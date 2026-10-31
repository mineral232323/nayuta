package com.nayuta.ime.converter.dict

import android.content.Context
import android.content.res.AssetManager
import android.util.Log
import java.io.BufferedInputStream
import java.io.BufferedReader
import java.io.FileNotFoundException
import java.io.InputStreamReader
import java.util.zip.GZIPInputStream

/**
 * カタカナ外来語 → 英単語の対応表（「あんどろいど → Android」）。
 *
 * JMdict から生成した読み(ひらがな) → 英単語 の表を読み込むだけ。
 * 変換時は対応する読みにだけ英語候補を足すので、通常の日本語変換には影響しない。
 */
class EnglishLexicon private constructor(
    private val map: HashMap<String, String>,
) {
    val size: Int get() = map.size

    fun lookup(reading: String): String? = map[reading]

    companion object {
        private const val ASSET_GZ = "dict/english.tsv.gz"
        private const val ASSET = "dict/english.tsv"

        @Volatile
        private var cached: EnglishLexicon? = null

        fun getOrNull(context: Context): EnglishLexicon? {
            cached?.let { return it }
            synchronized(this) {
                cached?.let { return it }
                return try {
                    val t0 = System.currentTimeMillis()
                    val lex = load(context)
                    Log.d("NayutaEnglish", "loaded entries=${lex.size} in ${System.currentTimeMillis() - t0}ms")
                    cached = lex
                    lex
                } catch (e: Throwable) {
                    Log.e("NayutaEnglish", "failed to load; english conversion disabled", e)
                    null
                }
            }
        }

        private fun open(context: Context): BufferedInputStream {
            val am: AssetManager = context.assets
            for (name in arrayOf(ASSET_GZ, ASSET)) {
                try {
                    return BufferedInputStream(am.open(name, AssetManager.ACCESS_STREAMING), 1 shl 16)
                } catch (_: FileNotFoundException) {
                    // 次の候補へ
                }
            }
            throw FileNotFoundException("dict/english.tsv(.gz) not packaged")
        }

        private fun load(context: Context): EnglishLexicon {
            val src = open(context)
            src.mark(2)
            val b0 = src.read()
            val b1 = src.read()
            src.reset()
            val body = if (b0 == 0x1f && b1 == 0x8b) GZIPInputStream(src, 1 shl 16) else src
            val map = HashMap<String, String>(1 shl 15)
            body.use {
                BufferedReader(InputStreamReader(it, Charsets.UTF_8), 1 shl 16).use { r ->
                    r.readLine() // ヘッダ
                    while (true) {
                        val line = r.readLine() ?: break
                        if (line.isEmpty() || line[0] == '#') continue
                        val t = line.indexOf('\t')
                        if (t <= 0) continue
                        map[line.substring(0, t)] = line.substring(t + 1)
                    }
                }
            }
            return EnglishLexicon(map)
        }
    }
}
