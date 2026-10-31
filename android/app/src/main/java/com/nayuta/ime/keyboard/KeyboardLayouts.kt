package com.nayuta.ime.keyboard

/**
 * キー配列の定義テーブル。Gboard の 12 キーに寄せた **5 列 × 4 行** 構成。
 *
 * かなモード:
 * ```
 *  小゛    あ  か  さ   ⌫
 *  ◀       た  な  は   ▶
 *  、      ま  や  ら   変換(空白)
 *  あa1   ー  わ  。   ✓
 * ```
 * - 左右の細い列が補助キー。`た` の左に左矢印、`は` の右に右矢印を置いている。
 * - `小゛` は **直前の 1 文字を変換する**キー。文字の種類で濁点か小文字かを決める
 *   （か→が、あ→ぁ、つ→っ）。長押しで半濁点（は→ぱ）。
 * - `変換(空白)` は 1 キー。未確定なら変換、確定済みなら半角スペース。
 *
 * 母音は **中央=あ / 左=い / 上=う / 右=え / 下=お**（市販のフリック入力と同じ）:
 * ```
 *        う
 *   い ← あ → え
 *        お
 * ```
 * 配列を変えたいときはこの表だけを触ればよい。描画側は配列知識を持たない。
 */
object KeyboardLayouts {

    /** 左右の補助キーの幅比（中央のキーを 1.0 としたとき）。 */
    private const val AUX = 0.72f

    private fun gojuon(label: String, i: String, u: String, e: String, o: String): FlickKey =
        FlickKey(
            label = label,
            center = label,
            up = u,
            right = e,
            left = i,
            down = o,
        )

    private fun aux(label: String, action: KeyAction, weight: Float = AUX): FunctionKey =
        FunctionKey(label = label, action = action, widthWeight = weight)

    /** 直前の文字を小文字化/濁点化するキー。長押しで半濁点。 */
    /**
     * 直前の文字を小文字/濁点に変えるキー。中央グリッド（ま の下）に置くので
     * 幅は通常キーと同じ 1.0（補助列の AUX ではない）。
     */
    private val transformKey = FunctionKey(
        label = "小゛",
        action = KeyAction.TRANSFORM,
        widthWeight = 1f,
        longPressAction = KeyAction.HANDAKUTEN,
        // 同じキーを短時間に 2 回 → 半濁点（は → ば → ぱ）
        doubleTapAction = KeyAction.HANDAKUTEN,
    )

    /** 変換と空白を兼ねるキー（未確定なら変換、確定済みなら半角スペース）。 */
    private val convertOrSpaceKey = FunctionKey(
        label = "変換",
        action = KeyAction.SPACE,
        widthWeight = AUX,
    )

    /**
     * 「、」と「。」を統合した句読点キー。
     * 中央グリッド（わ の右）に置くので幅は通常キーと同じ 1.0。
     */
    private val commaKey = FlickKey(
        label = "、。",
        center = "、",
        up = "。",
        right = "？",
        left = "！",
        down = "・",
        widthWeight = 1f,
    )

    private val longVowelKey = FlickKey(
        label = "ー",
        center = "ー",
        up = "〜",
        right = "…",
        left = "「",
        down = "」",
    )

    // ---------------- かな ----------------

    private val kanaRows: List<KeyboardRow> = listOf(
        KeyboardRow(
            listOf(
                // 変換候補の訂正（機能は未実装・予約枠）
                FunctionKey(
                    label = "訂正",
                    action = KeyAction.CORRECT,
                    widthWeight = AUX,
                ),
                gojuon("あ", "い", "う", "え", "お"),
                gojuon("か", "き", "く", "け", "こ"),
                gojuon("さ", "し", "す", "せ", "そ"),
                aux("⌫", KeyAction.DELETE),
            ),
        ),
        KeyboardRow(
            listOf(
                aux("◀", KeyAction.CURSOR_LEFT),
                gojuon("た", "ち", "つ", "て", "と"),
                gojuon("な", "に", "ぬ", "ね", "の"),
                gojuon("は", "ひ", "ふ", "へ", "ほ"),
                aux("▶", KeyAction.CURSOR_RIGHT),
            ),
        ),
        KeyboardRow(
            listOf(
                aux("数字", KeyAction.NUMBER_TOGGLE),
                gojuon("ま", "み", "む", "め", "も"),
                // や行は や・ゆ・よ のみ。ゆ=う段なので上、よ=お段なので下。
                FlickKey(label = "や", center = "や", up = "ゆ", down = "よ"),
                gojuon("ら", "り", "る", "れ", "ろ"),
                convertOrSpaceKey,
            ),
        ),
        KeyboardRow(
            listOf(
                aux("あa1", KeyAction.MODE_TOGGLE),
                transformKey,
                // わ行: を=お段なので下、ん・ー は残りへ。
                FlickKey(label = "わ", center = "わ", up = "ん", right = "ー", down = "を"),
                commaKey,
                aux("✓", KeyAction.ENTER),
            ),
        ),
    )

    // ---------------- 数字テンキー ----------------

    /** 数字テンキーの補助列は 0.5 キー幅（数字を広く取る）。 */
    private const val NUM_AUX = 0.5f

    private fun numAux(label: String, action: KeyAction): FunctionKey =
        FunctionKey(label = label, action = action, widthWeight = NUM_AUX)

    private fun digit(label: String): FlickKey = FlickKey(label = label, center = label)

    private val numberRows: List<KeyboardRow> = listOf(
        KeyboardRow(
            listOf(
                numAux("あa1", KeyAction.KANA_TOGGLE),
                digit("1"),
                digit("2"),
                digit("3"),
                // 「3 の隣」に削除
                numAux("⌫", KeyAction.DELETE),
            ),
        ),
        KeyboardRow(
            listOf(
                numAux("◀", KeyAction.CURSOR_LEFT),
                digit("4"),
                digit("5"),
                digit("6"),
                numAux("▶", KeyAction.CURSOR_RIGHT),
            ),
        ),
        KeyboardRow(
            listOf(
                // ピリオドは 0.5 キー
                FlickKey(
                    label = ".",
                    center = ".",
                    up = "-",
                    left = "/",
                    down = ":",
                    widthWeight = NUM_AUX,
                ),
                digit("7"),
                digit("8"),
                digit("9"),
                // カンマも 0.5 キー
                FlickKey(
                    label = ",",
                    center = ",",
                    up = "+",
                    left = "=",
                    down = "\\",
                    widthWeight = NUM_AUX,
                ),
            ),
        ),
        KeyboardRow(
            listOf(
                FlickKey(
                    label = "ー",
                    center = "ー",
                    up = "〜",
                    right = "…",
                    widthWeight = NUM_AUX,
                ),
                // かっこ。タップで「（」、上フリックで「）」
                FlickKey(
                    label = "（）",
                    center = "（",
                    up = "）",
                    right = "「",
                    left = "」",
                ),
                digit("0"),
                FlickKey(label = "00", center = "00"),
                numAux("✓", KeyAction.ENTER),
            ),
        ),
    )

    // ---------------- 英数（QWERTY） ----------------

    /**
     * QWERTY 配列。各行の合計幅を 10 に揃えて、キー幅が行をまたいで揃うようにしている。
     * 2 段目の左右は [SpacerKey] でずらす。
     */
    fun alphabet(shift: Boolean): KeyboardLayout {
        fun letter(c: Char): FlickKey {
            val s = if (shift) c.uppercaseChar() else c
            return FlickKey(label = s.toString(), center = s.toString())
        }

        val row0 = "qwertyuiop".map { letter(it) }
        val row1 = listOf(SpacerKey(0.5f)) +
            "asdfghjkl".map { letter(it) } +
            listOf(SpacerKey(0.5f))
        val row2 = listOf(
            FunctionKey(
                label = if (shift) "⇧" else "⇧",
                action = KeyAction.SHIFT,
                widthWeight = 1.5f,
            ),
        ) + "zxcvbnm".map { letter(it) } + listOf(
            FunctionKey(label = "⌫", action = KeyAction.DELETE, widthWeight = 1.5f),
        )
        // 下段: 123 / , / 空白 / . / あa1 / ✓ （合計 10）
        val row3 = listOf(
            FunctionKey(label = "123", action = KeyAction.NUMBER_TOGGLE, widthWeight = 1.5f),
            FlickKey(label = ",", center = ",", widthWeight = 1.0f),
            FunctionKey(label = "空白", action = KeyAction.SPACE, widthWeight = 3.0f),
            FlickKey(label = ".", center = ".", widthWeight = 1.0f),
            FunctionKey(label = "あa1", action = KeyAction.MODE_TOGGLE, widthWeight = 1.5f),
            FunctionKey(label = "✓", action = KeyAction.ENTER, widthWeight = 2.0f),
        )

        return KeyboardLayout(
            listOf(
                KeyboardRow(row0),
                KeyboardRow(row1),
                KeyboardRow(row2),
                KeyboardRow(row3),
            ),
        )
    }

    fun forMode(mode: KeyboardMode): KeyboardLayout = when (mode) {
        KeyboardMode.KANA -> KeyboardLayout(kanaRows)
        KeyboardMode.ALPHABET -> alphabet(shift = false)
        KeyboardMode.NUMBER -> KeyboardLayout(numberRows)
    }
}
