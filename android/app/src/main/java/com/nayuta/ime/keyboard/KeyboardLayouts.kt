package com.nayuta.ime.keyboard

/**
 * キー配列の定義テーブル。Gboard の 12 キーに寄せた **5 列 × 4 行** 構成。
 *
 * かなモード:
 * ```
 *  1@      あ  か  さ   ⌫
 *  ◀       た  な  は   ▶
 *  数字    ま  や  ら   変換(空白)
 *  あa     小゛ わ  、。  ✓
 * ```
 * - 左右の細い列が補助キー。`た` の左に左矢印、`は` の右に右矢印を置いている。
 * - `小゛` は直前の 1 文字を変換するキー。Gboard と同じく
 *   中央/下 = 小文字 → 濁点 → 半濁点 を巡回、上 = 小文字、左 = 濁点、右 = 半濁点。
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

    /** 英字・記号ページの行の高さ倍率（かなは 1.0）。 */
    private const val ALPHA_HEIGHT_SCALE = 0.82f

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

    /**
     * カーソル移動キー。タップは左右、フリックは上下左右に動く（Gboard と同じ）。
     */
    private fun cursorKey(label: String, action: KeyAction, weight: Float = AUX): FunctionKey =
        FunctionKey(
            label = label,
            action = action,
            widthWeight = weight,
            flickActions = mapOf(
                FlickDirection.UP to KeyAction.CURSOR_UP,
                FlickDirection.DOWN to KeyAction.CURSOR_DOWN,
                FlickDirection.LEFT to KeyAction.CURSOR_LEFT,
                FlickDirection.RIGHT to KeyAction.CURSOR_RIGHT,
            ),
        )

    /**
     * 直前の文字を小文字/濁点/半濁点に変えるキー。中央グリッド（ま の下）に置くので
     * 幅は通常キーと同じ 1.0（補助列の AUX ではない）。
     *
     * Gboard と同じ割り当て:
     *   中央 = 文字種に応じて 小文字 → 濁点 → 半濁点 を巡回
     *   上   = 小文字
     *   左   = 濁点
     *   右   = 半濁点
     *   下   = 中央と同じ
     */
    private val transformKey = FunctionKey(
        label = "小゛",
        action = KeyAction.TRANSFORM,
        widthWeight = 1f,
        flickActions = mapOf(
            FlickDirection.UP to KeyAction.SMALL,
            FlickDirection.LEFT to KeyAction.DAKUTEN,
            FlickDirection.RIGHT to KeyAction.HANDAKUTEN,
        ),
        flickLabels = mapOf(
            FlickDirection.UP to "小",
            FlickDirection.LEFT to "゛",
            FlickDirection.RIGHT to "゜",
        ),
    )

    /** 変換と空白を兼ねるキー（未確定なら変換、確定済みなら半角スペース）。 */
    private val convertOrSpaceKey = FunctionKey(
        label = "変換",
        action = KeyAction.SPACE,
        widthWeight = AUX,
    )

    /**
     * 句読点キー（Gboard と同じ配置）。
     *
     *         ？
     *    。  、  ！
     *         …
     *
     * 中央をダブルタップすると「。」。
     */
    private val commaKey = FlickKey(
        label = "、。",
        center = "、",
        up = "？",
        right = "！",
        left = "。",
        down = "…",
        widthWeight = 1f,
        doubleTapText = "。",
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
                // 数字・記号ページを呼び出す
                FunctionKey(
                    label = "1@",
                    action = KeyAction.SYMBOL_TOGGLE,
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
                cursorKey("◀", KeyAction.CURSOR_LEFT),
                gojuon("た", "ち", "つ", "て", "と"),
                gojuon("な", "に", "ぬ", "ね", "の"),
                gojuon("は", "ひ", "ふ", "へ", "ほ"),
                cursorKey("▶", KeyAction.CURSOR_RIGHT),
            ),
        ),
        KeyboardRow(
            listOf(
                aux("数字", KeyAction.NUMBER_TOGGLE),
                gojuon("ま", "み", "む", "め", "も"),
                // や行は や・ゆ・よ のみ。左右は Gboard と同じく括弧。
                FlickKey(
                    label = "や",
                    center = "や",
                    up = "ゆ",
                    down = "よ",
                    left = "（",
                    right = "）",
                ),
                gojuon("ら", "り", "る", "れ", "ろ"),
                convertOrSpaceKey,
            ),
        ),
        KeyboardRow(
            listOf(
                aux("あa", KeyAction.MODE_TOGGLE),
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
                numAux("あa", KeyAction.KANA_TOGGLE),
                digit("1"),
                digit("2"),
                digit("3"),
                // 「3 の隣」に削除
                numAux("⌫", KeyAction.DELETE),
            ),
        ),
        KeyboardRow(
            listOf(
                cursorKey("◀", KeyAction.CURSOR_LEFT, NUM_AUX),
                digit("4"),
                digit("5"),
                digit("6"),
                cursorKey("▶", KeyAction.CURSOR_RIGHT, NUM_AUX),
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
    fun alphabet(shift: Boolean, numberRow: Boolean = true): KeyboardLayout {
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
        // 下段: あa / , / 空白 / . / 123 / ✓ （合計 10）
        val row3 = listOf(
            FunctionKey(label = "あa", action = KeyAction.MODE_TOGGLE, widthWeight = 1.5f),
            FlickKey(label = ",", center = ",", widthWeight = 1.0f),
            FunctionKey(label = "空白", action = KeyAction.SPACE, widthWeight = 3.0f),
            FlickKey(label = ".", center = ".", widthWeight = 1.0f),
            FunctionKey(label = "123", action = KeyAction.SYMBOL_TOGGLE, widthWeight = 1.5f),
            FunctionKey(label = "✓", action = KeyAction.ENTER, widthWeight = 2.0f),
        )

        val rows = mutableListOf<KeyboardRow>()
        if (numberRow) {
            // 数字行（キー幅は他の英字キーと同じ）
            rows.add(KeyboardRow("1234567890".map { symbolKey(it.toString()) }))
        }
        rows.add(KeyboardRow(row0))
        rows.add(KeyboardRow(row1))
        rows.add(KeyboardRow(row2))
        rows.add(KeyboardRow(row3))

        return KeyboardLayout(
            rows,
            // 英字はかなより 1 行あたりの高さを低くして、画面を広く使えるようにする
            heightScale = ALPHA_HEIGHT_SCALE,
        )
    }

    // ---------------- 数字＋記号（英字の 123） ----------------
    //
    //  1 2 3 4 5 6 7 8 9 0
    //  @ # ¥ % & - + ( ) /
    //  ABC * " ' : ; ! ?  ⌫
    //  あa  ,   空白   .   ⏎
    private fun symbolKey(label: String): FlickKey = FlickKey(label = label, center = label)

    private val symbolRows: List<KeyboardRow> = listOf(
        KeyboardRow("1234567890".map { symbolKey(it.toString()) }),
        KeyboardRow("@#¥%&-+()/".map { symbolKey(it.toString()) }),
        KeyboardRow(
            listOf(
                FunctionKey(label = "ABC", action = KeyAction.ALPHABET_TOGGLE, widthWeight = 1.5f),
                symbolKey("*"),
                symbolKey("\""),
                symbolKey("'"),
                symbolKey(":"),
                symbolKey(";"),
                symbolKey("!"),
                symbolKey("?"),
                FunctionKey(label = "⌫", action = KeyAction.DELETE, widthWeight = 1.5f),
            ),
        ),
        KeyboardRow(
            listOf(
                FunctionKey(label = "あa", action = KeyAction.KANA_TOGGLE, widthWeight = 1.7f),
                symbolKey(","),
                FunctionKey(label = "空白", action = KeyAction.SPACE, widthWeight = 5.0f),
                symbolKey("."),
                FunctionKey(label = "⏎", action = KeyAction.ENTER, widthWeight = 1.7f),
            ),
        ),
    )

    fun forMode(mode: KeyboardMode): KeyboardLayout = when (mode) {
        KeyboardMode.KANA -> KeyboardLayout(kanaRows)
        KeyboardMode.ALPHABET -> alphabet(shift = false)
        KeyboardMode.NUMBER -> KeyboardLayout(numberRows)
        KeyboardMode.SYMBOL -> KeyboardLayout(symbolRows, heightScale = ALPHA_HEIGHT_SCALE)
    }
}
