package com.nayuta.ime.converter.dict

import android.content.Context
import android.util.Log
import com.nayuta.ime.converter.Converter

/**
 * 単語ラティス + Viterbi によるかな漢字変換。
 *
 * 読みを左から走査して「辞書に載っている読み」の全スパンを弧として並べ、
 * 最小コストの経路を選ぶ。コストは現状 unigram（単語頻度）のみで、
 * 連接コストや文脈スコアは載せていない。
 *
 * 文脈（context）は今は使っていない。BERT 採点を載せるときに、
 * ここで各弧へスコアを足す（`Arc.extra` を用意してある）。
 */
class LatticeConverter(
    private val appContext: Context,
) : Converter {

    @Volatile
    private var dict: Dictionary? = null

    /** 重いロードは初回変換時に、呼び出し側のスレッドで 1 回だけ行う。 */
    private fun dictionary(): Dictionary {
        dict?.let { return it }
        synchronized(this) {
            dict?.let { return it }
            val t0 = System.currentTimeMillis()
            val d = Dictionary.get(appContext)
            Log.d("NayutaDict", "loaded readings=${d.readingCount} entries=${d.entryTotal} maxLen=${d.maxReadingLength} in ${System.currentTimeMillis() - t0}ms")
            dict = d
            return d
        }
    }

    override fun convert(reading: String, context: String): List<String> {
        if (reading.isEmpty()) return emptyList()
        val d = dictionary()

        val n = reading.length
        val byStart = Array(n) { ArrayList<Arc>() }

        // 辞書スパン
        var maxLen = d.maxReadingLength
        for (i in 0 until n) {
            val limit = minOf(n, i + maxLen)
            var j = i + 1
            while (j <= limit) {
                val idx = d.indexOfReading(reading, i, j)
                if (idx >= 0) {
                    val cnt = d.surfaceCountOf(idx)
                    for (k in 0 until cnt) {
                        byStart[i].add(
                            Arc(i, j, d.surfaceOf(idx, k), d.costOf(idx, k), 0)
                        )
                    }
                }
                j++
            }
        }
        // 未登録文字は「そのままかな」で通す（高コスト）
        for (i in 0 until n) {
            byStart[i].add(
                Arc(i, i + 1, reading.substring(i, i + 1), d.unknownCost, 0)
            )
        }

        val paths = nbest(n, byStart, beam = 8, want = 12)
        Log.d("NayutaConv", "reading=$reading paths=${paths.size} best=${paths.firstOrNull()?.let { p -> p.joinToString("") { it.surface } }}")
        if (paths.isEmpty()) return listOf(reading)

        val out = LinkedHashSet<String>()
        for (p in paths) {
            val s = buildString { for (a in p) append(a.surface) }
            out.add(s)
            if (out.size >= 9) break
        }
        return out.toList()
    }

    // ------------------------------------------------------------------

    private class Arc(
        val start: Int,
        val end: Int,
        val surface: String,
        val cost: Int,
        /** BERT 採点を足すときの拡張点（今は常に 0）。 */
        val extra: Int,
    ) {
        val total: Int get() = cost + extra
    }

    private class Hyp(val cost: Int, val arcs: List<Arc>)

    /** ビーム探索で上位 [want] 件の経路を返す。 */
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
}
