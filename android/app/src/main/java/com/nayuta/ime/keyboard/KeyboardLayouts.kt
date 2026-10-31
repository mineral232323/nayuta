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
    private val transformKey = FunctionKey(
        label = "小゛",
        action = KeyAction.TRANSFORM,
        widthWeight = AUX,
        longPressAction = KeyAction.HANDAKUTEN,
    )

    /** 変換と空白を兼ねるキー（未確定なら変換、確定済みなら半角スペース）。 */
    private val convertOrSpaceKey = FunctionKey(
        label = "変換",
        action = KeyAction.SPACE,
        widthWeight = AUX,
    )

    private val commaKey = FlickKey(
        label = "、",
        center = "、",
        up = "。",
        right = "？",
        left = "！",
        down = "・",
        // 左の補助列に置くので、他の補助キーと同じ幅にする（列がずれないように）
        widthWeight = AUX,
    )

    private val periodKey = FlickKey(
        label = "。",
        center = "。",
        up = "、",
        right = "）",
        left = "（",
        down = "・",
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
                transformKey,
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
                commaKey,
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
                longVowelKey,
                // わ行: を=お段なので下、ん・ー は残りへ。
                FlickKey(label = "わ", center = "わ", up = "ん", right = "ー", down = "を"),
                periodKey,
                aux("✓", KeyAction.ENTER),
            ),
        ),
    )

    // ---------------- 英数 ----------------

    private fun alphaKey(label: String, center: String, up: String?, right: String?): FlickKey =
        FlickKey(label = label, center = center, up = up, right = right)

    private val alphabetRows: List<KeyboardRow> = listOf(
        KeyboardRow(
            listOf(
                FlickKey(
                    label = ",",
                    center = ",",
                    up = ";",
                    right = ":",
                    left = "?",
                    down = "!",
                    widthWeight = AUX,
                ),
                alphaKey("ABC", "a", "b", "c"),
                alphaKey("DEF", "d", "e", "f"),
                alphaKey("GHI", "g", "h", "i"),
                aux("⌫", KeyAction.DELETE),
            ),
        ),
        KeyboardRow(
            listOf(
                aux("◀", KeyAction.CURSOR_LEFT),
                alphaKey("JKL", "j", "k", "l"),
                alphaKey("MNO", "m", "n", "o"),
                alphaKey("PQR", "p", "q", "r"),
                aux("▶", KeyAction.CURSOR_RIGHT),
            ),
        ),
        KeyboardRow(
            listOf(
                FlickKey(
                    label = ".",
                    center = ".",
                    up = "-",
                    down = "/",
                    widthWeight = AUX,
                ),
                alphaKey("STU", "s", "t", "u"),
                alphaKey("VWX", "v", "w", "x"),
                alphaKey("YZ", "y", "z", null),
                convertOrSpaceKey,
            ),
        ),
        KeyboardRow(
            listOf(
                aux("aA1", KeyAction.MODE_TOGGLE),
                FlickKey(label = "@", center = "@", up = "_", right = "#"),
                FlickKey(label = "_", center = "_", up = "=", right = "+"),
                FlickKey(label = "/", center = "/", up = "\\", right = "|"),
                aux("✓", KeyAction.ENTER),
            ),
        ),
    )

    fun forMode(mode: KeyboardMode): KeyboardLayout = when (mode) {
        KeyboardMode.KANA -> KeyboardLayout(kanaRows)
        KeyboardMode.ALPHABET -> KeyboardLayout(alphabetRows)
    }
}
