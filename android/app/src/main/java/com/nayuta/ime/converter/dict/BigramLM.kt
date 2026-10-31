package com.nayuta.ime.converter.dict

import android.content.Context
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * 単語 bigram の連接コスト。
 *
 * 「直前の単語 → 次の単語」の生起コストを持ち、未観測の連接は null を返す
 * （呼び出し側は unigram コストにフォールバックする）。
 *
 * データは CSR 形式: starts[prev] .. starts[prev+1] の範囲に rights/costs が並ぶ。
 */
class BigramLM private constructor(
    private val wordId: HashMap<String, Int>,
    private val starts: IntArray,
    private val rights: IntArray,
    private val costs: ShortArray,
) {

    val words: Int get() = starts.size - 1
    val pairs: Int get() = rights.size

    /** 連接コスト（1/100 nat 単位）。未観測なら null。 */
    fun cost(prev: String?, cur: String): Int? {
        if (prev == null) return null
        val p = wordId[prev] ?: return null
        val c = wordId[cur] ?: return null
        var lo = starts[p]
        var hi = starts[p + 1] - 1
        while (lo <= hi) {
            val mid = (lo + hi) ushr 1
            val r = rights[mid]
            when {
                r < c -> lo = mid + 1
                r > c -> hi = mid - 1
                else -> return costs[mid].toInt()
            }
        }
        return null
    }

    companion object {
        private const val ASSET = "dict/bigram.bin"
        private const val MAGIC = 0x5947424E

        @Volatile
        private var cached: BigramLM? = null

        fun getOrNull(context: Context): BigramLM? {
            cached?.let { return it }
            synchronized(this) {
                cached?.let { return it }
                return try {
                    val t0 = System.currentTimeMillis()
                    val lm = load(context)
                    Log.d(
                        "NayutaBigram",
                        "loaded words=${lm.words} pairs=${lm.pairs} in ${System.currentTimeMillis() - t0}ms",
                    )
                    cached = lm
                    lm
                } catch (e: Throwable) {
                    Log.e("NayutaBigram", "failed to load bigram; bigram disabled", e)
                    null
                }
            }
        }

        private fun load(context: Context): BigramLM {
            val bytes = context.assets.open(ASSET).use { it.readBytes() }
            val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            val magic = buf.int
            if (magic != MAGIC) throw IllegalStateException("bad magic: $magic")
            buf.int // version
            val v = buf.int
            val p = buf.int

            val starts = IntArray(v + 1)
            for (i in 0..v) starts[i] = buf.int
            val rights = IntArray(p)
            buf.asIntBuffer().get(rights)
            buf.position(buf.position() + 4 * p)
            val costs = ShortArray(p)
            buf.asShortBuffer().get(costs)
            buf.position(buf.position() + 2 * p)

            val textEnd = bytes.size
            val textLen = textEnd - buf.position()
            val text = String(bytes, buf.position(), textLen, Charsets.UTF_8)
            val lines = text.split('\n')
            val map = HashMap<String, Int>(v * 2)
            for (i in 0 until v) {
                if (i < lines.size && lines[i].isNotEmpty()) map[lines[i]] = i
            }
            return BigramLM(map, starts, rights, costs)
        }
    }
}
