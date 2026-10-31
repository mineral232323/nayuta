package com.nayuta.ime.converter.dict

import android.content.Context
import android.util.Log
import com.nayuta.ime.converter.Converter
import com.nayuta.ime.converter.bert.BertScorer

/**
 * 単語ラティス + Viterbi + BERT 再ランクによるかな漢字変換。
 *
 * 2 段構え:
 *   1. 辞書 unigram コストでビーム探索し、上位 [WANT] 本の変換仮説を作る
 *   2. それらに出てくるスパンだけを BERT で採点し、`辞書コスト + β·(−log P)` で並べ直す
 *
 * 全スパンを採点すると端末では重いので、探索で絞ったスパンだけを 1 回の forward で見る。
 * 文脈（直前の確定済みテキスト）は BERT の入力にだけ使う。
 */
class LatticeConverter(
    private val appContext: Context,
) : Converter {

    @Volatile
    private var dict: Dictionary? = null

    @Volatile
    private var scorer: BertScorer? = null

    @Volatile
    private var scorerFailed = false

    @Volatile
    private var userDict: UserDictionary? = null

    @Volatile
    private var bigram: BigramLM? = null

    @Volatile
    private var bigramTried = false

    private fun dictionary(): Dictionary {
        dict?.let { return it }
        synchronized(this) {
            dict?.let { return it }
            val t0 = System.currentTimeMillis()
            val d = Dictionary.get(appContext)
            Log.d(
                "NayutaDict",
                "loaded readings=${d.readingCount} entries=${d.entryTotal} in ${System.currentTimeMillis() - t0}ms",
            )
            dict = d
            return d
        }
    }

    /** BERT が無い/壊れている場合は辞書のみで動かす（品質は落ちるが止まらない）。 */
    private fun scorerOrNull(): BertScorer? {
        if (scorerFailed) return null
        scorer?.let { return it }
        synchronized(this) {
            scorer?.let { return it }
            if (scorerFailed) return null
            return try {
                val t0 = System.currentTimeMillis()
                val s = BertScorer(appContext)
                Log.d("NayutaBert", "loaded in ${System.currentTimeMillis() - t0}ms")
                scorer = s
                s
            } catch (e: Throwable) {
                Log.e("NayutaBert", "failed to load scorer; falling back to unigram", e)
                scorerFailed = true
                null
            }
        }
    }

    /** 訂正パネルで登録したユーザー辞書。 */
    private fun userDictionary(): UserDictionary =
        userDict ?: synchronized(this) {
            userDict ?: UserDictionary.get(appContext).also { userDict = it }
        }

    /** 連接コスト。無くても動く（unigram のみに戻る）。 */
    private fun bigramOrNull(): BigramLM? {
        if (bigramTried) return bigram
        synchronized(this) {
            if (bigramTried) return bigram
            bigram = BigramLM.getOrNull(appContext)
            bigramTried = true
            return bigram
        }
    }

    override fun convert(reading: String, context: String): List<String> {
        if (reading.isEmpty()) return emptyList()
        val d = dictionary()
        val n = reading.length
        val byStart = Array(n) { ArrayList<Arc>() }

        val maxLen = minOf(d.maxReadingLength, MAX_SPAN_LEN)
        for (i in 0 until n) {
            val limit = minOf(n, i + maxLen)
            var j = i + 1
            while (j <= limit) {
                val idx = d.indexOfReading(reading, i, j)
                if (idx >= 0) {
                    val cnt = minOf(d.surfaceCountOf(idx), MAX_CANDIDATES)
                    for (k in 0 until cnt) {
                        byStart[i].add(Arc(i, j, d.surfaceOf(idx, k), d.costOf(idx, k), 0))
                    }
                }
                // ユーザー辞書は最優先（コスト 0 で必ず勝つ）
                val userSurfaces = userDictionary().entries(reading.substring(i, j))
                if (userSurfaces.isNotEmpty()) {
                    for (u in userSurfaces) byStart[i].add(Arc(i, j, u, USER_COST, 0))
                }
                j++
            }
            byStart[i].add(Arc(i, i + 1, reading.substring(i, i + 1), d.unknownCost, 0))
        }

        val t0 = System.currentTimeMillis()
        val paths = nbest(n, byStart, beam = BEAM, want = WANT)
        if (paths.isEmpty()) return listOf(reading)
        val searchMs = System.currentTimeMillis() - t0

        val s = scorerOrNull()
        val ranked = if (s != null) {
            try {
                rerank(s, context, reading, paths)
            } catch (e: Throwable) {
                Log.e("NayutaBert", "rerank failed", e)
                paths
            }
        } else {
            paths
        }

        val out = LinkedHashSet<String>()
        for (p in ranked) {
            out.add(buildString { for (a in p) append(a.surface) })
            if (out.size >= 9) break
        }
        Log.d(
            "NayutaConv",
            "reading=$reading best=${out.firstOrNull()} paths=${paths.size} " +
                "search=${searchMs}ms total=${System.currentTimeMillis() - t0}ms",
        )
        return out.toList()
    }

    // ------------------------------------------------------------------

    private class Arc(
        val start: Int,
        val end: Int,
        val surface: String,
        val cost: Int,
        val extra: Int,
    ) {
        val total: Int get() = cost + extra
    }

    private class Hyp(val cost: Int, val arcs: List<Arc>)

    private fun nbest(
        n: Int,
        byStart: Array<ArrayList<Arc>>,
        beam: Int,
        want: Int,
    ): List<List<Arc>> {
        val beams = Array(n + 1) { ArrayList<Hyp>() }
        beams[0].add(Hyp(0, emptyList()))
        for (i in 0 until n) {
            val cur = beams[i]
            if (cur.isEmpty()) continue
            cur.sortBy { it.cost }
            val kept = if (cur.size > beam) cur.subList(0, beam) else cur
            for (h in kept) {
                for (arc in byStart[i]) {
                    beams[arc.end].add(Hyp(h.cost + arc.total, h.arcs + arc))
                }
            }
        }
        val last = beams[n]
        if (last.isEmpty()) return emptyList()
        last.sortBy { it.cost }
        return last.take(want).map { it.arcs }
    }

    /**
     * 探索で得た仮説に出てくるスパンだけを BERT で採点して並べ直す。
     * 仮説に無いスパンは採点しない（端末の計算量を抑えるため）。
     */
    private fun rerank(
        s: BertScorer,
        context: String,
        reading: String,
        paths: List<List<Arc>>,
    ): List<List<Arc>> {
        val spanSurfaces = LinkedHashMap<Long, MutableList<String>>()
        for (path in paths) {
            for (a in path) {
                val key = a.start.toLong() * 1000L + a.end
                val list = spanSurfaces.getOrPut(key) { ArrayList() }
                if (list.size < MAX_CANDIDATES && !list.contains(a.surface)) {
                    list.add(a.surface)
                }
            }
        }
        val keys = spanSurfaces.keys.toList()
        val reqs = keys.map { key ->
            val st = (key / 1000L).toInt()
            val en = (key % 1000L).toInt()
            BertScorer.SpanRequest(st, en, spanSurfaces[key]!!)
        }
        val lps = s.score(context, reading, reqs, MAX_CANDIDATES)

        val logp = HashMap<Long, HashMap<String, Float>>(keys.size)
        for ((i, key) in keys.withIndex()) {
            val m = HashMap<String, Float>(4)
            val surfaces = spanSurfaces[key]!!
            for (j in surfaces.indices) m[surfaces[j]] = lps[i][j]
            logp[key] = m
        }

        val lm = bigramOrNull()
        val scored = ArrayList<Pair<Int, List<Arc>>>(paths.size)
        for (path in paths) {
            var c = 0
            var prev: String? = null
            for (a in path) {
                // 連接コストが観測済みなら unigram の代わりに使う（分割の誤りを減らす）
                c += lm?.cost(prev, a.surface) ?: a.cost
                val key = a.start.toLong() * 1000L + a.end
                val lp = logp[key]?.get(a.surface) ?: WORST_LOGPROB
                c += (-BETA * lp * 100f).toInt()
                prev = a.surface
            }
            scored.add(Pair(c, path))
        }
        scored.sortBy { it.first }
        return scored.map { it.second }
    }

    companion object {
        private const val BEAM = 32
        private const val WANT = 48
        private const val MAX_SPAN_LEN = 16
        private const val MAX_CANDIDATES = BertScorer.MAX_CANDIDATES

        /** BERT スコアの重み。辞書コストは 1/100 nat 単位なので -β·logP·100 を足す。 */
        private const val BETA = 1.2f

        /** ユーザー辞書の候補コスト。0 なので通常の辞書より必ず優先される。 */
        private const val USER_COST = 0

        /** 仮説に候補が無かったときの代替 logP。 */
        private const val WORST_LOGPROB = -6.0f
    }
}
