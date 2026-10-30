package com.nayuta.ime.converter

/**
 * スタブ実装。今は読みをひらがなのまま 1 件だけ返す。
 *
 * 辞書もモデルも無い段階で UI とサービスを通しで動かすためのもので、
 * 変換キーを押すと「読み」がそのまま候補として出る。
 *
 * TODO(M2): 単語ラティス + ONNX BERT 採点に差し替える。
 *          差し替え先は [Converters.create] の一行だけ。
 */
class KanaConverter : Converter {

    override fun convert(reading: String, context: String): List<String> {
        if (reading.isEmpty()) return emptyList()
        // 「今は入力をそのままひらがなで返すだけ」
        return listOf(reading)
    }
}
