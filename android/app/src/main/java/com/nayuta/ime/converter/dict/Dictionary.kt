package com.nayuta.ime.converter.dict

import android.content.Context
import android.content.res.AssetManager
import java.io.BufferedInputStream
import java.io.BufferedReader
import java.io.FileInputStream
import java.io.FileNotFoundException
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.zip.GZIPInputStream

/**
 * 読み → 表記候補の辞書。
 *
 * 端末用アセットは `assets/dict/lexicon.ndict`（パック済みバイナリ）。
 * APK 内で非圧縮にしておけば **mmap で読むだけ**で、Java ヒープをほぼ使わない。
 * 旧アセット（`dict/lexicon.tsv.gz`）しか無い場合は、読み込んで同じレイアウトの
 * ダイレクトバッファに詰め直す（フォールバック）。
 *
 * レイアウト（little endian。すべて 4 バイト境界に整列）:
 * ```
 * 0  magic "NDX2" / 4 version / 8 readingCount / 12 entryCount / 16 maxReadingLen
 * 20 readingCharsOff / 24 readingCharsLen（UTF-16 コード単位）
 * 28 surfaceCharsOff / 32 surfaceCharsLen
 * 36 readingOffsetsOff（rc+1 個の u32）/ 40 entryStartOff（rc 個の u32）
 * 44 entryCountOff（rc 個の u16）/ 48 surfaceOffsetsOff（ec+1 個の u32）
 * 52 costsOff（ec 個の i16 = -logP*100）
 * ```
 */
class Dictionary private constructor(
    private val buf: ByteBuffer,
    private val readingCountValue: Int,
    private val entryTotalValue: Int,
    /** 読みの最大文字数（ラティスのスパン長上限）。 */
    val maxReadingLength: Int,
    private val rCharOff: Int,
    private val sCharOff: Int,
    private val rOffOff: Int,
    private val eStartOff: Int,
    private val eCountOff: Int,
    private val sOffOff: Int,
    private val costOff: Int,
) {

    /** 未登録1文字のコスト（辞書に無い読みをそのまま通す）。 */
    val unknownCost: Int = 2500

    val readingCount: Int get() = readingCountValue
    val entryTotal: Int get() = entryTotalValue

    fun surfaceCountOf(readingIndex: Int): Int =
        buf.getShort(eCountOff + readingIndex * 2).toInt()

    fun surfaceOf(readingIndex: Int, k: Int): String {
        val idx = buf.getInt(eStartOff + readingIndex * 4) + k
        val start = buf.getInt(sOffOff + idx * 4)
        val end = buf.getInt(sOffOff + (idx + 1) * 4)
        val chars = CharArray(end - start)
        var p = sCharOff + start * 2
        for (i in chars.indices) {
            chars[i] = buf.getChar(p)
            p += 2
        }
        return String(chars)
    }

    fun costOf(readingIndex: Int, k: Int): Int {
        val idx = buf.getInt(eStartOff + readingIndex * 4) + k
        return buf.getShort(costOff + idx * 2).toInt()
    }

    /** 読みを二分探索する。見つからなければ -1。 */
    fun indexOfReading(text: String, from: Int, to: Int): Int {
        var lo = 0
        var hi = readingCountValue - 1
        while (lo <= hi) {
            val mid = (lo + hi) ushr 1
            val c = compareRange(mid, text, from, to)
            when {
                c < 0 -> lo = mid + 1
                c > 0 -> hi = mid - 1
                else -> return mid
            }
        }
        return -1
    }

    /** readings[mid] と text[from, to) を比較する（部分文字列も String も作らない）。 */
    private fun compareRange(idx: Int, text: String, from: Int, to: Int): Int {
        val start = buf.getInt(rOffOff + idx * 4)
        val end = buf.getInt(rOffOff + (idx + 1) * 4)
        val len = end - start
        val bLen = to - from
        val n = minOf(len, bLen)
        var p = rCharOff + start * 2
        for (i in 0 until n) {
            val d = buf.getChar(p).code - text[from + i].code
            if (d != 0) return d
            p += 2
        }
        return len - bLen
    }

    companion object {
        private const val ASSET_BIN = "dict/lexicon.ndict"
        private const val ASSET_TSV_GZ = "dict/lexicon.tsv.gz"
        private const val ASSET_TSV = "dict/lexicon.tsv"
        private const val MAGIC = 0x3258444E // "NDX2" little endian
        private const val VERSION = 1
        private const val HEADER = 64

        @Volatile
        private var cached: Dictionary? = null

        fun get(context: Context): Dictionary =
            cached ?: synchronized(this) {
                cached ?: load(context.applicationContext).also { cached = it }
            }

        fun load(context: Context): Dictionary {
            try {
                return fromBuffer(mmapAsset(context, ASSET_BIN))
            } catch (e: Throwable) {
                // .ndict が無い / 圧縮されている場合は旧 TSV から組み立てる
                android.util.Log.d(
                    "NayutaDict",
                    "ndict unavailable (${e.javaClass.simpleName}); falling back to tsv",
                )
            }
            return fromTsv(context)
        }

        /** 非圧縮アセットを mmap する（APK 内のページをそのまま参照する）。 */
        private fun mmapAsset(context: Context, name: String): ByteBuffer {
            val afd = context.assets.openFd(name)
            return afd.use {
                FileInputStream(it.fileDescriptor).use { stream ->
                    stream.channel
                        .map(
                            FileChannel.MapMode.READ_ONLY,
                            it.startOffset,
                            it.declaredLength,
                        )
                        .order(ByteOrder.LITTLE_ENDIAN)
                }
            }
        }

        private fun fromBuffer(raw: ByteBuffer): Dictionary {
            val buf = raw.order(ByteOrder.LITTLE_ENDIAN)
            if (buf.getInt(0) != MAGIC) throw IllegalStateException("bad dictionary magic")
            if (buf.getInt(4) != VERSION) throw IllegalStateException("bad dictionary version")
            val rc = buf.getInt(8)
            val ec = buf.getInt(12)
            return Dictionary(
                buf = buf,
                readingCountValue = rc,
                entryTotalValue = ec,
                maxReadingLength = buf.getInt(16),
                rCharOff = buf.getInt(20),
                sCharOff = buf.getInt(28),
                rOffOff = buf.getInt(36),
                eStartOff = buf.getInt(40),
                eCountOff = buf.getInt(44),
                sOffOff = buf.getInt(48),
                costOff = buf.getInt(52),
            )
        }

        // ------------------------------------------------------------------
        // フォールバック: 旧 TSV(.gz) をパースして同じレイアウトに詰める
        // ------------------------------------------------------------------

        private fun openAsset(context: Context): BufferedInputStream {
            val am = context.assets
            for (name in arrayOf(ASSET_TSV_GZ, ASSET_TSV)) {
                try {
                    return BufferedInputStream(
                        am.open(name, AssetManager.ACCESS_STREAMING),
                        1 shl 16,
                    )
                } catch (_: FileNotFoundException) {
                    // 次の候補へ
                }
            }
            throw FileNotFoundException("dict/lexicon.ndict(.tsv.gz) not packaged")
        }

        private fun fromTsv(context: Context): Dictionary {
            val readings = ArrayList<String>(1 shl 18)
            val starts = ArrayList<Int>(1 shl 18)
            val counts = ArrayList<Int>(1 shl 18)
            val surfaces = ArrayList<String>(1 shl 19)
            val costs = ArrayList<Int>(1 shl 19)
            var entryCount = 0

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
                            readings.add(reading)
                            starts.add(entryCount)
                            counts.add(0)
                            prevReading = reading
                        }
                        counts[counts.size - 1]++
                        surfaces.add(surface)
                        costs.add(cost)
                        entryCount++
                    }
                }
            }
            return pack(readings, surfaces, costs, starts, counts)
        }

        private fun pack(
            readings: List<String>,
            surfaces: List<String>,
            costs: List<Int>,
            starts: List<Int>,
            counts: List<Int>,
        ): Dictionary {
            val rc = readings.size
            val ec = surfaces.size
            val rLen = readings.sumOf { it.length }
            val sLen = surfaces.sumOf { it.length }
            fun align4(x: Int): Int = (x + 3) and 3.inv()
            var off = HEADER
            val rCharOff = off
            off = align4(off + rLen * 2)
            val sCharOff = off
            off = align4(off + sLen * 2)
            val rOffOff = off
            off += 4 * (rc + 1)
            val sOffOff = off
            off += 4 * (ec + 1)
            val eStartOff = off
            off += 4 * rc
            val eCountOff = off
            off += 2 * rc
            val costOff = off
            off += 2 * ec

            val buf = ByteBuffer.allocateDirect(off).order(ByteOrder.LITTLE_ENDIAN)
            buf.putInt(0, MAGIC)
            buf.putInt(4, VERSION)
            buf.putInt(8, rc)
            buf.putInt(12, ec)
            buf.putInt(16, readings.maxOfOrNull { it.length } ?: 1)
            buf.putInt(20, rCharOff)
            buf.putInt(24, rLen)
            buf.putInt(28, sCharOff)
            buf.putInt(32, sLen)
            buf.putInt(36, rOffOff)
            buf.putInt(40, eStartOff)
            buf.putInt(44, eCountOff)
            buf.putInt(48, sOffOff)
            buf.putInt(52, costOff)

            var p = rCharOff
            for (rd in readings) {
                for (ch in rd) {
                    buf.putChar(p, ch)
                    p += 2
                }
            }
            p = sCharOff
            for (s in surfaces) {
                for (ch in s) {
                    buf.putChar(p, ch)
                    p += 2
                }
            }
            var acc = 0
            for (i in 0 until rc) {
                buf.putInt(rOffOff + i * 4, acc)
                acc += readings[i].length
            }
            buf.putInt(rOffOff + rc * 4, acc)
            acc = 0
            for (i in 0 until ec) {
                buf.putInt(sOffOff + i * 4, acc)
                acc += surfaces[i].length
            }
            buf.putInt(sOffOff + ec * 4, acc)
            for (i in 0 until rc) {
                buf.putInt(eStartOff + i * 4, starts[i])
                buf.putShort(eCountOff + i * 2, counts[i].toShort())
            }
            for (i in 0 until ec) {
                val c = costs[i]
                if (c < Short.MIN_VALUE || c > Short.MAX_VALUE) {
                    throw IllegalStateException("cost out of i16 range: $c")
                }
                buf.putShort(costOff + i * 2, c.toShort())
            }
            return fromBuffer(buf)
        }
    }
}
