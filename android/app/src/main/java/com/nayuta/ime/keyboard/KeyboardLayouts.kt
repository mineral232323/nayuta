package com.nayuta.ime.keyboard

/**
 * 12 キー配列の定義テーブル。
 *
 * かなモード（日本語フリック配列）:
 * ```
 *   あ(あ い う え お)   か(か き く け こ)   さ(さ し す せ そ)
 *   た(た ち つ て と)   な(な に ぬ ね の)   は(は ひ ふ へ ほ)
 *   ま(ま み む め も)   や(や ゆ よ)         ら(ら り る れ ろ)
 *   わ(わ を ん ー)      小(っ ゃ ゅ ょ ぁ)   ゛(゛ ゜ 、 。 ・)
 *   [英数] [空白] [変換] [削除]
 * ```
 *
 * フリック方向は母音の並びに合わせて **左=い / 上=う / 右=え / 下=お**:
 * ```
 *        う
 *   い ← あ → え
 *        お
 * ```
 * つまり 中央=あ / 左=い / 上=う / 右=え / 下=お。
 * や行は「や・ゆ・よ」しか無いので未使用方向は null（無入力）。
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

    /**
     * 5 母音の行（あ〜ら行）を作る。引数は母音順（あ, い, う, え, お）で、
     * 配置は **中央=あ / 左=い / 上=う / 右=え / 下=お**。
     * 引数の順番を覚えればよいので、行ごとに方向を書き間違えない。
     */
    private fun gojuon(label: String, i: String, u: String, e: String, o: String): FlickKey =
        FlickKey(
            label = label,
            center = label,
            up = u,
            right = e,
            left = i,
            down = o,
        )

    /** 母音列ではないキー（や行・わ行・小・゛）用に方向を明示して作る。 */
    private fun flick(
        label: String,
        center: String = label,
        up: String? = null,
        right: String? = null,
        left: String? = null,
        down: String? = null,
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
                gojuon("あ", "い", "う", "え", "お"),
                gojuon("か", "き", "く", "け", "こ"),
                gojuon("さ", "し", "す", "せ", "そ"),
            ),
        ),
        KeyboardRow(
            listOf(
                gojuon("た", "ち", "つ", "て", "と"),
                gojuon("な", "に", "ぬ", "ね", "の"),
                gojuon("は", "ひ", "ふ", "へ", "ほ"),
            ),
        ),
        KeyboardRow(
            listOf(
                gojuon("ま", "み", "む", "め", "も"),
                // や行は や・ゆ・よ のみ。ゆ=う段 -> 上、よ=お段 -> 下。
                flick("や", up = "ゆ", down = "よ"),
                gojuon("ら", "り", "る", "れ", "ろ"),
            ),
        ),
        KeyboardRow(
            listOf(
                // わ行: を=お段なので下。ん・ー は残りの方向へ。
                flick("わ", up = "ん", right = "ー", down = "を"),
                // 小: タップで「っ」、ゆ=う段 -> 上、よ=お段 -> 下、ゃ -> 左、ぁ -> 右。
                flick("小", center = "っ", up = "ゅ", right = "ぁ", left = "ゃ", down = "ょ"),
                // ゛゜: タップで濁点、上フリックで半濁点、左右下は句読点・中黒
                flick(
                    label = KanaMarks.DAKUTEN,
                    center = KanaMarks.DAKUTEN,
                    up = KanaMarks.HANDAKUTEN,
                    right = "、",
                    left = "。",
                    down = "・",
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
        heightWeight = 0.96f,
    )

    // ---------------- 英数 ----------------

    private val alphabetRows: List<KeyboardRow> = listOf(
        KeyboardRow(
            listOf(
                flick("ABC", center = "a", up = "b", right = "c"),
                flick("DEF", center = "d", up = "e", right = "f"),
                flick("GHI", center = "g", up = "h", right = "i"),
            ),
        ),
        KeyboardRow(
            listOf(
                flick("JKL", center = "j", up = "k", right = "l"),
                flick("MNO", center = "m", up = "n", right = "o"),
                flick("PQR", center = "p", up = "q", right = "r"),
            ),
        ),
        KeyboardRow(
            listOf(
                flick("STU", center = "s", up = "t", right = "u"),
                flick("VWX", center = "v", up = "w", right = "x"),
                flick("YZ", center = "y", up = "z"),
            ),
        ),
        KeyboardRow(
            keys = listOf(
                FunctionKey("かな", KeyAction.MODE_TOGGLE),
                FunctionKey("空白", KeyAction.SPACE),
                FunctionKey("削除", KeyAction.DELETE),
            ),
            heightWeight = 0.96f,
        ),
    )

    fun forMode(mode: KeyboardMode): KeyboardLayout = when (mode) {
        KeyboardMode.KANA -> KeyboardLayout(kanaRows)
        KeyboardMode.ALPHABET -> KeyboardLayout(alphabetRows)
    }
}
