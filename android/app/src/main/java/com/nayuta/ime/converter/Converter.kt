package com.nayuta.ime.converter

/**
 * Nayuta の変換エンジンと Android 側の唯一の接点。
 *
 * 設計書 docs/design.md §4 の「①候補生成 (単語ラティス + Viterbi) → ②BERT 採点 →
 * ③再ランク」は、すべてこの 1 メソッドの向こう側に隠す。UI 側 (service / view) は
 * `convert()` が候補文字列のリストを返すことだけを知っていればよい。
 *
 * 実装差し替えは [Converters.create] の 1 箇所だけで済む。
 *
 * ## 契約
 * - [reading] はひらがなのみの読み列（例: `"きょうはいいてんき"`）。
 * - [context] は変換対象より前の文脈文字列（最大 64 文字程度、無ければ空文字）。
 *   文脈は「直前の確定済みテキスト」で、読み自体は含まない。
 * - 戻り値は**優先順の候補リスト**。空リストは「候補なし」を意味し、
 *   UI 側は読みをそのまま表示・確定する。
 * - 呼び出しはメインスレッド外（単一スレッドの executor）から行われる前提。
 *   実装はブロッキングしてよい（ONNX 推論をそのまま書ける）。
 */
interface Converter {
    fun convert(reading: String, context: String): List<String>

    /**
     * 読みを単語単位に区切り、各語の候補を返す（単語ごとの変換モード用）。
     *
     * `convert()` が「文章全体の候補」を返すのに対し、こちらは
     * 「どこで区切るか」と「各語の候補」を返す。UI 側は選択中の語の候補だけを出す。
     */
    fun segments(reading: String, context: String): List<Segment>

    /** 読み 1 語分の表記候補（区切りを手動で変えたときの引き直しに使う）。 */
    fun candidatesFor(reading: String): List<String>

    /**
     * 確定した内容から「薄い層」をオンライン更新する。
     * 対応していない実装は何もしなくてよい。
     */
    fun learn(context: String, reading: String, candidates: List<String>, chosen: Int) {}
}

/**
 * 単語 1 つ分。[start] / [end] は読み文字列の添字。
 * [candidates] はその語の表記候補（優先順）。
 */
data class Segment(
    val start: Int,
    val end: Int,
    val candidates: List<String>,
)
