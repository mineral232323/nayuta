package com.nayuta.ime.converter

import android.content.Context
import com.nayuta.ime.converter.dict.LatticeConverter

/**
 * 変換エンジンの生成点。**ここが差し替え点の唯一の場所**。
 *
 * 今は [KanaConverter]（読みをそのまま返すスタブ）を返す。将来
 * 「単語ラティス + ONNX の BERT 採点」を実装したら、[create] の中の一行を
 * その実装に置き換えるだけでよい:
 *
 * ```kotlin
 * private fun create(context: Context): Converter =
 *     LatticeBertConverter(
 *         dictionary = Dictionary.load(context.assets, "dict/system.dic"),
 *         model = BertScorer.load(context.assets, "model/nayuta_char_bert.onnx"),
 *     )
 * ```
 *
 * [Context] を渡してあるのは、その実装が assets から辞書と ONNX モデルを
 * 読む必要があるため。スタブは context を使わない。
 */
object Converters {

    @Volatile
    private var instance: Converter? = null

    /** プロセス内で 1 個だけ生成する（モデルのロードは高価なので）。 */
    fun get(context: Context): Converter =
        instance ?: synchronized(this) {
            instance ?: create(context.applicationContext).also { instance = it }
        }

    // ======================= 差し替え点 =======================
    private fun create(context: Context): Converter {
        // M2: 単語ラティス + Viterbi（unigram）。BERT 採点はこの後ろに載せる。
        // 辞書のロードは初回 convert() 時に呼び出しスレッドで行われる。
        return LatticeConverter(context)
    }
    // ==========================================================
}
