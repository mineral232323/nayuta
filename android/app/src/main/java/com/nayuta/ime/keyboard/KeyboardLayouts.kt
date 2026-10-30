package com.nayuta.ime.keyboard

/**
 * 12 キー配列の定義テーブル。
 *
 * かなモード（標準的な日本語フリック配列）:
 * ```
 *   あ(あ い う え お)   か(か き く け こ)   さ(さ し す せ そ)
 *   た(た ち つ て と)   な(な に ぬ ね の)   は(は ひ ふ へ ほ)
 *   ま(ま み む め も)   や(や ゆ よ)         ら(ら り る れ ろ)
 *   わ(わ を ん ー 、)   小(っ ゃ ゅ ょ ぁ)   ゛(゛ ゜ 、 。 ・)
 *   [英数] [空白] [変換] [削除]
 * ```
 * フリック方向は 上=2 番目 / 右=3 番目 / 左=4 番目 / 下=5 番目。
 * や行は「や・ゆ・よ」しか無いので上下左右のうち未使用方向は null（無入力）。
 *
 * 英数モード（電話機の 12 キー英字配列）:
 * ```
 *   ABC  DEF  GHI
 *   JKL  MNO  PQR
 *   STU  VWX  YZ
 *   [かな] [空白] [削除]
 * ```
 * タップで 1 文字目、上=2 文字目、右=3 文字目、左=4 文字目。
 *
 * この表を書き換えるだけで配列を変えられるよう、描画側には一切配列知識を置かない。
 */
object KeyboardLayouts {

    private fun flick(
        label: String,
        up: String? = null,
        right: String? = null,
        left: String? = null,
        down: String? = null,
        center: String = label,
    ): FlickKey = FlickKey(
        label = label,
        center = center,
        up = up,
        right = right,
        left = left,
        down = down,
    )

    // ---------------- かな ----------------

    private val kanaRows: List<KeyboardRow> = listOf(
        KeyboardRow(
            listOf(
                flick("あ", "い", "う", "え", "お"),
                flick("か", "き", "く", "け", "こ"),
                flick("さ", "し", "す", "せ", "そ"),
            ),
        ),
        KeyboardRow(
            listOf(
                flick("た", "ち", "つ", "て", "と"),
                flick("な", "に", "ぬ", "ね", "の"),
                flick("は", "ひ", "ふ", "へ", "ほ"),
            ),
        ),
        KeyboardRow(
            listOf(
                flick("ま", "み", "む", "め", "も"),
                flick("や", right = "ゆ", down = "よ"),
                flick("ら", "り", "る", "れ", "ろ"),
            ),
        ),
        KeyboardRow(
            listOf(
                flick("わ", "を", "ん", "ー", "、"),
                // 小: タップで「っ」、フリックで小書きかな
                flick("小", up = "ゃ", right = "ゅ", left = "ょ", down = "ぁ", center = "っ"),
                // ゛゜: タップで濁点、上フリックで半濁点、左右下は句読点
                flick(
                    label = KanaMarks.DAKUTEN,
                    up = KanaMarks.HANDAKUTEN,
                    right = "、",
                    left = "。",
                    down = "・",
                    center = KanaMarks.DAKUTEN,
                ),
            ),
        ),
        kanaFunctionRow(),
    )

    private fun kanaFunctionRow(): KeyboardRow = KeyboardRow(
        keys = listOf(
            FunctionKey("英数", KeyAction.MODE_TOGGLE),
            FunctionKey("空白", KeyAction.SPACE),
            FunctionKey("変換", KeyAction.CONVERT),
            FunctionKey("削除", KeyAction.DELETE),
        ),
        heightWeight = 0.78f,
    )

    // ---------------- 英数 ----------------

    private val alphabetRows: List<KeyboardRow> = listOf(
        KeyboardRow(
            listOf(
                flick("ABC", up = "b", right = "c", center = "a"),
                flick("DEF", up = "e", right = "f", center = "d"),
                flick("GHI", up = "h", right = "i", center = "g"),
            ),
        ),
        KeyboardRow(
            listOf(
                flick("JKL", up = "k", right = "l", center = "j"),
                flick("MNO", up = "n", right = "o", center = "m"),
                flick("PQR", up = "q", right = "r", center = "p"),
            ),
        ),
        KeyboardRow(
            listOf(
                flick("STU", up = "t", right = "u", center = "s"),
                flick("VWX", up = "w", right = "x", center = "v"),
                flick("YZ", up = "z", center = "y"),
            ),
        ),
        KeyboardRow(
            keys = listOf(
                FunctionKey("かな", KeyAction.MODE_TOGGLE),
                FunctionKey("空白", KeyAction.SPACE),
                FunctionKey("削除", KeyAction.DELETE),
            ),
            heightWeight = 0.78f,
        ),
    )

    fun forMode(mode: KeyboardMode): KeyboardLayout = when (mode) {
        KeyboardMode.KANA -> KeyboardLayout(kanaRows)
        KeyboardMode.ALPHABET -> KeyboardLayout(alphabetRows)
    }
}
