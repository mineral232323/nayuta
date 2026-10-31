package com.nayuta.ime.keyboard

/**
 * キー配列の定義テーブル。Gboard の 12 キーに寄せた **5 列 × 4 行** 構成。
 *
 * ```
 *  ◀    あ  か  さ   ⌫
 *  ▶    た  な  は   ゛
 *  小   ま  や  ら   変換
 *  あa1 空白 わ  ?!  ✓
 * ```
 * 左右の細い列が補助キー、中央 3 列がかなの 12 キー。
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

    private val smallKey = FlickKey(
        label = "小",
        center = "っ",
        up = "ゅ",
        right = "ぁ",
        left = "ゃ",
        down = "ょ",
        widthWeight = AUX,
    )

    private val dakutenKey = FlickKey(
        label = KanaMarks.DAKUTEN,
        center = KanaMarks.DAKUTEN,
        up = KanaMarks.HANDAKUTEN,
        right = "、",
        left = "。",
        down = "・",
        widthWeight = AUX,
    )

    private val symbolKey = FlickKey(
        label = "?!",
        center = "、",
        up = "。",
        right = "？",
        left = "！",
        down = "・",
    )

    // ---------------- かな ----------------

    private val kanaRows: List<KeyboardRow> = listOf(
        KeyboardRow(
            listOf(
                aux("◀", KeyAction.CURSOR_LEFT),
                gojuon("あ", "い", "う", "え", "お"),
                gojuon("か", "き", "く", "け", "こ"),
                gojuon("さ", "し", "す", "せ", "そ"),
                aux("⌫", KeyAction.DELETE),
            ),
        ),
        KeyboardRow(
            listOf(
                aux("▶", KeyAction.CURSOR_RIGHT),
                gojuon("た", "ち", "つ", "て", "と"),
                gojuon("な", "に", "ぬ", "ね", "の"),
                gojuon("は", "ひ", "ふ", "へ", "ほ"),
                dakutenKey,
            ),
        ),
        KeyboardRow(
            listOf(
                smallKey,
                gojuon("ま", "み", "む", "め", "も"),
                // や行は や・ゆ・よ のみ。ゆ=う段なので上、よ=お段なので下。
                FlickKey(label = "や", center = "や", up = "ゆ", down = "よ"),
                gojuon("ら", "り", "る", "れ", "ろ"),
                aux("変換", KeyAction.CONVERT),
            ),
        ),
        KeyboardRow(
            listOf(
                aux("あa1", KeyAction.MODE_TOGGLE),
                aux("空白", KeyAction.SPACE, weight = 1f),
                // わ行: を=お段なので下、ん・ー は残りへ。
                FlickKey(label = "わ", center = "わ", up = "ん", right = "ー", down = "を"),
                symbolKey,
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
                aux("◀", KeyAction.CURSOR_LEFT),
                alphaKey("ABC", "a", "b", "c"),
                alphaKey("DEF", "d", "e", "f"),
                alphaKey("GHI", "g", "h", "i"),
                aux("⌫", KeyAction.DELETE),
            ),
        ),
        KeyboardRow(
            listOf(
                aux("▶", KeyAction.CURSOR_RIGHT),
                alphaKey("JKL", "j", "k", "l"),
                alphaKey("MNO", "m", "n", "o"),
                alphaKey("PQR", "p", "q", "r"),
                aux("、", KeyAction.CONVERT),
            ),
        ),
        KeyboardRow(
            listOf(
                aux("。", KeyAction.CONVERT),
                alphaKey("STU", "s", "t", "u"),
                alphaKey("VWX", "v", "w", "x"),
                alphaKey("YZ", "y", "z", null),
                aux("変換", KeyAction.CONVERT),
            ),
        ),
        KeyboardRow(
            listOf(
                aux("aA1", KeyAction.MODE_TOGGLE),
                aux("空白", KeyAction.SPACE, weight = 1f),
                FlickKey(label = "@", center = "@", up = "_", right = "-", down = "/"),
                symbolKey,
                aux("✓", KeyAction.ENTER),
            ),
        ),
    )

    fun forMode(mode: KeyboardMode): KeyboardLayout = when (mode) {
        KeyboardMode.KANA -> KeyboardLayout(kanaRows)
        KeyboardMode.ALPHABET -> KeyboardLayout(alphabetRows)
    }
}
