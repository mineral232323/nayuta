package com.nayuta.ime.keyboard

/** フリック方向。標準的な 12 キー日本語配列では CENTER=母音 a / i / u / e / o。 */
enum class FlickDirection {
    CENTER,
    UP,
    RIGHT,
    LEFT,
    DOWN,
}

/**
 * フリック入力できるキー。`label` はキー上に大きく出す表示、
 * CENTER 以外の出力はサブ（上=2 番目, 右=3 番目, 左=4 番目, 下=5 番目）に対応する。
 *
 * 例: あ行 -> CENTER あ / UP い / RIGHT う / LEFT え / DOWN お
 */
data class FlickKey(
    override val label: String,
    val center: String = label,
    val up: String? = null,
    val right: String? = null,
    val left: String? = null,
    val down: String? = null,
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
) : KeySpec()

enum class KeyAction {
    DELETE,
    CONVERT,
    SPACE,
    MODE_TOGGLE,
}

/** キー 1 個の抽象。 */
sealed class KeySpec {
    abstract val label: String
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
    ALPHABET;

    fun toggled(): KeyboardMode = if (this == KANA) ALPHABET else KANA
}
