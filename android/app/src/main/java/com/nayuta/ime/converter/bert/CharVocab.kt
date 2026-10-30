package com.nayuta.ime.converter.bert

import android.content.Context
import java.text.Normalizer

/**
 * 文字 -> BERT の入力 id。
 *
 * モデル側 (train/scorer.py の CharVocab) と同じ規則:
 *   - 1 文字ずつ NFKC 正規化し、1 文字に収まる場合だけその文字で引く
 *   - 語彙に無ければ [UNK]
 *
 * 読み・文脈ともこの規則で id 化するので、文字数と id 数が 1:1 で対応する。
 */
class CharVocab private constructor(private val map: HashMap<String, Int>) {

    val size: Int = map.size
    val unk: Long = (map["[UNK]"] ?: 0).toLong()
    val cls: Long = (map["[CLS]"] ?: 1).toLong()
    val sep: Long = (map["[SEP]"] ?: 2).toLong()

    fun encode(text: String): LongArray {
        val out = LongArray(text.length)
        for (i in text.indices) {
            val ch = text[i]
            val n = Normalizer.normalize(ch.toString(), Normalizer.Form.NFKC)
            val key = if (n.length == 1) n else ch.toString()
            out[i] = (map[key] ?: unk.toInt()).toLong()
        }
        return out
    }

    companion object {
        private const val ASSET = "model/vocab.txt"

        @Volatile
        private var cached: CharVocab? = null

        fun get(context: Context): CharVocab =
            cached ?: synchronized(this) {
                cached ?: load(context).also { cached = it }
            }

        fun load(context: Context): CharVocab {
            val map = HashMap<String, Int>(8192)
            context.assets.open(ASSET).bufferedReader(Charsets.UTF_8).useLines { lines ->
                lines.forEachIndexed { i, line -> map[line] = i }
            }
            return CharVocab(map)
        }
    }
}
