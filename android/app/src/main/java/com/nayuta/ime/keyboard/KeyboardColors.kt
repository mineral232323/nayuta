package com.nayuta.ime.keyboard

import android.content.Context
import android.content.res.Configuration
import android.graphics.Color

/**
 * Gboard 風 / Material 3 の配色トークン。ライト・ダークの 2 パレットを持つ。
 *
 * 呼び出し側は `KeyboardColors.key` のようにプロパティを読むだけでよく、
 * 現在どちらのテーマかは [setDark] で切り替える。色を保持したままの View は
 * 各自 `applyTheme()` を実装して再適用する（Paint の color は生成時に確定するため）。
 */
object KeyboardColors {

    /** 1 テーマ分の色。 */
    data class Palette(
        val keyboardBackground: Int,
        val stripBackground: Int,
        val key: Int,
        val keyPressed: Int,
        val functionKey: Int,
        val functionKeyPressed: Int,
        val keyText: Int,
        val keySubText: Int,
        val keySubTextActive: Int,
        val balloonBackground: Int,
        val balloonText: Int,
        val balloonOutline: Int,
        val balloonShadow: Int,
        val candidateText: Int,
        val candidateSelectedText: Int,
        val candidateSelectedBackground: Int,
        val divider: Int,
        val ripple: Int,
    )

    /**
     * ライト。surfaceContainer / surfaceContainerLowest / surfaceContainerHigh を
     * そのままキーの階層に対応させている。
     */
    private fun light() = Palette(
        keyboardBackground = Color.parseColor("#F1F3F4"),
        stripBackground = Color.parseColor("#F8F9FA"),
        key = Color.parseColor("#FFFFFF"),
        keyPressed = Color.parseColor("#E8F0FE"),
        functionKey = Color.parseColor("#E6E8EB"),
        functionKeyPressed = Color.parseColor("#D3D7DC"),
        keyText = Color.parseColor("#1B1C1E"),
        keySubText = Color.parseColor("#5F6368"),
        keySubTextActive = Color.parseColor("#0B57D0"),
        balloonBackground = Color.parseColor("#FFFFFF"),
        balloonText = Color.parseColor("#1B1C1E"),
        balloonOutline = Color.parseColor("#E3E6EA"),
        balloonShadow = Color.argb(7, 0, 0, 0),
        candidateText = Color.parseColor("#1B1C1E"),
        candidateSelectedText = Color.parseColor("#041E49"),
        candidateSelectedBackground = Color.parseColor("#D3E3FD"),
        divider = Color.parseColor("#E3E6EA"),
        ripple = Color.parseColor("#1F000000"),
    )

    /**
     * ダーク。Gboard のダークに寄せたニュートラル + M3 の primary 系。
     * キーは地より一段明るくして、面の階層が逆転しないようにする。
     */
    private fun dark() = Palette(
        keyboardBackground = Color.parseColor("#1E1F22"),
        stripBackground = Color.parseColor("#26272B"),
        key = Color.parseColor("#35363A"),
        keyPressed = Color.parseColor("#414A6B"),
        functionKey = Color.parseColor("#2A2B2F"),
        functionKeyPressed = Color.parseColor("#3A3C42"),
        keyText = Color.parseColor("#E3E3E3"),
        keySubText = Color.parseColor("#9AA0A6"),
        keySubTextActive = Color.parseColor("#A8C7FA"),
        balloonBackground = Color.parseColor("#3C4043"),
        balloonText = Color.parseColor("#E8EAED"),
        balloonOutline = Color.parseColor("#5F6368"),
        balloonShadow = Color.argb(64, 0, 0, 0),
        candidateText = Color.parseColor("#E3E3E3"),
        candidateSelectedText = Color.parseColor("#D3E3FD"),
        candidateSelectedBackground = Color.parseColor("#0842A0"),
        divider = Color.parseColor("#3C4043"),
        ripple = Color.parseColor("#1FFFFFFF"),
    )

    @Volatile
    private var palette: Palette = light()

    /** 現在のテーマがダークか。 */
    @Volatile
    var isDark: Boolean = false
        private set

    /**
     * テーマを決める。設定で「ライト/ダーク」が選ばれていればそれに従い、
     * 「システム」なら端末の uiMode に従う。
     */
    fun resolve(context: Context): Boolean {
        val mode = com.nayuta.ime.NayutaSettings.themeMode
        val dark = when (mode) {
            1 -> false
            2 -> true
            else -> {
                val night = context.resources.configuration.uiMode and
                    Configuration.UI_MODE_NIGHT_MASK
                night == Configuration.UI_MODE_NIGHT_YES
            }
        }
        setDark(dark)
        return dark
    }

    fun setDark(dark: Boolean) {
        isDark = dark
        palette = if (dark) dark() else light()
    }

    val keyboardBackground: Int get() = palette.keyboardBackground
    val stripBackground: Int get() = palette.stripBackground
    val key: Int get() = palette.key
    val keyPressed: Int get() = palette.keyPressed
    val functionKey: Int get() = palette.functionKey
    val functionKeyPressed: Int get() = palette.functionKeyPressed
    val keyText: Int get() = palette.keyText
    val keySubText: Int get() = palette.keySubText
    val keySubTextActive: Int get() = palette.keySubTextActive
    val balloonBackground: Int get() = palette.balloonBackground
    val balloonText: Int get() = palette.balloonText
    val balloonOutline: Int get() = palette.balloonOutline
    val balloonShadow: Int get() = palette.balloonShadow
    val candidateText: Int get() = palette.candidateText
    val candidateSelectedText: Int get() = palette.candidateSelectedText
    val candidateSelectedBackground: Int get() = palette.candidateSelectedBackground
    val divider: Int get() = palette.divider
    val ripple: Int get() = palette.ripple
}
