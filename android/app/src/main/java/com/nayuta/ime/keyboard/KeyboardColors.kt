package com.nayuta.ime.keyboard

import android.graphics.Color

/** キーボードの配色。ダークテーマ対応は M2 以降。 */
object KeyboardColors {
    val keyboardBackground: Int = Color.parseColor("#D9DEE7")
    val key: Int = Color.parseColor("#FFFFFF")
    val keyPressed: Int = Color.parseColor("#AFC3FF")
    val keyText: Int = Color.parseColor("#1B1B1F")
    val keySubText: Int = Color.parseColor("#5C5F66")
    val keySubTextActive: Int = Color.parseColor("#1D4ED8")
    val balloonBackground: Int = Color.parseColor("#2F3440")
    val balloonText: Int = Color.parseColor("#FFFFFF")

    val candidateBackground: Int = Color.parseColor("#F3F4F8")
    val candidateText: Int = Color.parseColor("#2B2F38")
    val candidateSelectedText: Int = Color.parseColor("#102A6B")
    val candidateSelectedBackground: Int = Color.parseColor("#D6E2FF")
}
