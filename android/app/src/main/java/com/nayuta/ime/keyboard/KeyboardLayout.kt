package com.nayuta.ime.keyboard

/**
 * フリック方向。
 *
 * 12 キーかな配列の母音は **中央=あ / 左=い / 上=う / 右=え / 下=お**:
 * ```
 *        う
 *   い ← あ → え
 *        お
 * ```
 */
enum class FlickDirection {
    CENTER,
    UP,
    RIGHT,
    LEFT,
    DOWN,
}

/**
 * フリック入力できるキー。`label` はキー上に大きく出す表示、
 * CENTER 以外の出力は母音の並び（左=い / 上=う / 右=え / 下=お）に対応する。
 *
 * 例: あ行 -> CENTER あ / LEFT い / UP う / RIGHT え / DOWN お
 */
data class FlickKey(
    override val label: String,
    val center: String = label,
    val up: String? = null,
    val right: String? = null,
    val left: String? = null,
    val down: String? = null,
    override val widthWeight: Float = 1f,
) : KeySpec() {

    /** 指定方向の出力。未定義方向は null（= 何も入力しない）。 */
    fun output(direction: FlickDirection): String? = when (direction) {
        FlickDirection.CENTER -> center
        FlickDirection.UP -> up
        FlickDirection.RIGHT -> right
        FlickDirection.LEFT -> left
        FlickDirection.DOWN -> down
    }

    /** 方向ごとのサブラベル（描画用）。CENTER は label を大きく出すので含めない。 */
    val subLabels: List<Pair<FlickDirection, String>>
        get() = buildList {
            up?.let { add(FlickDirection.UP to it) }
            right?.let { add(FlickDirection.RIGHT to it) }
            left?.let { add(FlickDirection.LEFT to it) }
            down?.let { add(FlickDirection.DOWN to it) }
        }
}

/** 入力ではなく動作を起こすキー（削除・変換・空白・英数切替）。 */
data class FunctionKey(
    override val label: String,
    val action: KeyAction,
    override val widthWeight: Float = 1f,
    /** 長押ししたときの動作（例: 小゛キーの長押しで半濁点）。 */
    val longPressAction: KeyAction? = null,
    /** 短時間に 2 回押したときの動作（例: 小゛キーのダブルタップで半濁点）。 */
    val doubleTapAction: KeyAction? = null,
) : KeySpec()

enum class KeyAction {
    DELETE,
    CONVERT,
    SPACE,
    MODE_TOGGLE,
    CURSOR_LEFT,
    CURSOR_RIGHT,
    ENTER,

    /** 直前の 1 文字を、文字種に応じて濁点化 or 小文字化する（トグル）。 */
    TRANSFORM,

    /** 直前の 1 文字を半濁点化する（小゛キーのダブルタップ/長押し用）。 */
    HANDAKUTEN,

    /** かな ⇄ 数字テンキー。 */
    NUMBER_TOGGLE,

    /** 数字テンキー → かな。 */
    KANA_TOGGLE,

    /** 変換候補の訂正（未実装・予約）。 */
    CORRECT,

    /** 英字の大文字/小文字切り替え。 */
    SHIFT,

    /** 英字 → 数字・記号（QWERTY 配列の記号ページ）。 */
    SYMBOL_TOGGLE,

    /** 数字・記号 → 英字。 */
    ALPHABET_TOGGLE,
}

/** キー 1 個の抽象。 */

/**
 * 何も描かない・何も入力しないスペーサー。QWERTY の段のずれを作るのに使う。
 */
data class SpacerKey(
    override val widthWeight: Float = 1f,
) : KeySpec() {
    override val label: String get() = ""
}

/** キー 1 個の抽象。`widthWeight` は行内での横幅比（左右の補助キーを細くする）。 */
sealed class KeySpec {
    abstract val label: String
    abstract val widthWeight: Float
}

/**
 * 単独の濁点・半濁点記号（U+309B / U+309C）。
 *
 * キーボードはこの文字をそのまま通知し、サービス側が直前のかなと合成する
 * （「か」+ ゛ -> 「が」）。リテラルを複数箇所に書くと文字コードがずれるので定数化する。
 */
object KanaMarks {
    const val DAKUTEN = "\u309B"
    const val HANDAKUTEN = "\u309C"
    const val DAKUTEN_CHAR = '\u309B'
    const val HANDAKUTEN_CHAR = '\u309C'
}

/**
 * 1 行分のキー列。`heightWeight` はキーボード内での行の高さ比
 * （かな行 = 1.0 に対して機能行を 0.78 などにする）。
 */
data class KeyboardRow(
    val keys: List<KeySpec>,
    val heightWeight: Float = 1f,
)

/** キーボード全体のレイアウト定義。描画は FlickKeyboardView が行う。 */
data class KeyboardLayout(val rows: List<KeyboardRow>)

/** かな入力と英数入力のモード。 */
enum class KeyboardMode {
    KANA,
    ALPHABET,
    /** かなの「数字」キーから入る 12 キーのテンキー。 */
    NUMBER,
    /** 英字の「123」キーから入る 数字＋記号のページ。 */
    SYMBOL;

    fun toggled(): KeyboardMode = when (this) {
        KANA -> ALPHABET
        ALPHABET -> KANA
        NUMBER -> KANA
        SYMBOL -> ALPHABET
    }
}
