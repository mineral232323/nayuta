package com.nayuta.ime.converter.dict

import android.content.Context
import android.content.res.AssetManager
import java.io.BufferedInputStream
import java.io.BufferedReader
import java.io.FileNotFoundException
import java.io.InputStreamReader
import java.util.zip.GZIPInputStream

/**
 * 読み → 表記候補の辞書。assets/dict/lexicon.tsv.gz を読み込む。
 *
 * ファイルは `読み \t 表記 \t コスト` の行を**読みの UTF-16 コード単位順**で並べたもの。
 * 読みが同じ行は連続するので、読みの配列 + 各読みの開始位置と件数だけ持てばよい。
 * コストは -log P を 100 倍した整数（小さいほど有利）。
 *
 * ロードは重い（数百 ms / 数十 MB）ので、必ずメインスレッド以外から呼ぶこと。
 */
class Dictionary private constructor(
    /** 重複を除いた読み（昇順）。 */
    private val readings: Array<String>,
    /** readings[i] の候補が始まる surfaces/costs の添字。 */
    private val entryStart: IntArray,
    /** readings[i] の候補数。 */
    private val entryCount: IntArray,
    private val surfaces: Array<String>,
    private val costs: IntArray,
) {

    /** 読みの最大文字数（ラティスのスパン長上限）。 */
    val maxReadingLength: Int = readings.maxOfOrNull { it.length } ?: 1

    /** 未登録1文字のコスト（辞書に無い読みをそのまま通す）。 */
    val unknownCost: Int = 2500

    val readingCount: Int get() = readings.size
    val entryTotal: Int get() = surfaces.size

    fun surfaceCountOf(readingIndex: Int): Int = entryCount[readingIndex]

    fun surfaceOf(readingIndex: Int, k: Int): String =
        surfaces[entryStart[readingIndex] + k]

    fun costOf(readingIndex: Int, k: Int): Int =
        costs[entryStart[readingIndex] + k]

    /** 読みを二分探索する。見つからなければ -1。 */
    fun indexOfReading(text: String, from: Int, to: Int): Int {
        var lo = 0
        var hi = readings.size - 1
        while (lo <= hi) {
            val mid = (lo + hi) ushr 1
            val c = compareRange(readings[mid], text, from, to)
            when {
                c < 0 -> lo = mid + 1
                c > 0 -> hi = mid - 1
                else -> return mid
            }
        }
        return -1
    }

    /** readings[mid] と text[from, to) を比較する（部分文字列を作らない）。 */
    private fun compareRange(a: String, text: String, from: Int, to: Int): Int {
        val bLen = to - from
        val n = minOf(a.length, bLen)
        for (i in 0 until n) {
            val d = a[i].code - text[from + i].code
            if (d != 0) return d
        }
        return a.length - bLen
    }

    companion object {
        /**
         * AAPT は `.gz` アセットを展開して同梱することがあるため、
         * 両方の名前を試し、さらに先頭 2 バイトで gzip かどうかを判定する。
         */
        private fun openAsset(context: Context): BufferedInputStream {
            val am = context.assets
            for (name in arrayOf("dict/lexicon.tsv.gz", "dict/lexicon.tsv")) {
                try {
                    return BufferedInputStream(
                        am.open(name, AssetManager.ACCESS_STREAMING),
                        1 shl 16,
                    )
                } catch (_: FileNotFoundException) {
                    // 次の候補へ
                }
            }
            throw FileNotFoundException("dict/lexicon.tsv(.gz) not packaged")
        }

        @Volatile
        private var cached: Dictionary? = null

        fun get(context: Context): Dictionary =
            cached ?: synchronized(this) {
                cached ?: load(context.applicationContext).also { cached = it }
            }

        fun load(context: Context): Dictionary {
            val readings = ArrayList<String>(1 shl 18)
            var starts = IntArray(1 shl 18)
            var counts = IntArray(1 shl 18)
            val surfaces = ArrayList<String>(1 shl 19)
            var costs = IntArray(1 shl 19)
            var nReadings = 0
            var nEntries = 0

            val src = openAsset(context)
            src.mark(2)
            val b0 = src.read()
            val b1 = src.read()
            src.reset()
            val body = if (b0 == 0x1f && b1 == 0x8b) GZIPInputStream(src, 1 shl 16) else src
            body.use { gz ->
                BufferedReader(InputStreamReader(gz, Charsets.UTF_8), 1 shl 16).use { r ->
                    r.readLine() // ヘッダ
                    var prevReading: String? = null
                    while (true) {
                        val line = r.readLine() ?: break
                        if (line.isEmpty()) continue
                        val t1 = line.indexOf('\t')
                        if (t1 <= 0) continue
                        val t2 = line.indexOf('\t', t1 + 1)
                        val reading = line.substring(0, t1)
                        val surface = if (t2 > 0) line.substring(t1 + 1, t2) else line.substring(t1 + 1)
                        val cost = if (t2 > 0) line.substring(t2 + 1).toIntOrNull() ?: 2000 else 2000
                        if (reading != prevReading) {
                            if (nReadings >= starts.size) {
                                starts = starts.copyOf(starts.size * 2)
                                counts = counts.copyOf(counts.size * 2)
                            }
                            readings.add(reading)
                            starts[nReadings] = nEntries
                            counts[nReadings] = 0
                            nReadings++
                            prevReading = reading
                        }
                        counts[nReadings - 1]++
                        surfaces.add(surface)
                        if (nEntries >= costs.size) costs = costs.copyOf(costs.size * 2)
                        costs[nEntries] = cost
                        nEntries++
                    }
                }
            }

            return Dictionary(
                readings = readings.toTypedArray(),
                entryStart = starts.copyOf(nReadings),
                entryCount = counts.copyOf(nReadings),
                surfaces = surfaces.toTypedArray(),
                costs = costs.copyOf(nEntries),
            )
        }
    }
}
