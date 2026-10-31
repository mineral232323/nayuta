package com.nayuta.ime.keyboard

import android.content.Context
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.widget.LinearLayout
import android.widget.TextView

/**
 * 訂正パネル。変換が外れた語をその場で直し、ユーザー辞書に登録する。
 *
 * ```
 *  読み  いーろんますく
 *  表記  イーロンマスク|
 *  [カタカナ] [ひらがな] [登録] [✕]
 * ```
 *
 * 表記はキーボードから直接編集できる（サービス側が入力をこのパネルへ振り分ける）。
 */
class CorrectionPanel @JvmOverloads constructor(
    context: Context,
    attrs: android.util.AttributeSet? = null,
) : LinearLayout(context, attrs) {

    /** 登録が押された。 */
    var onCommit: ((String) -> Unit)? = null

    /** 閉じるが押された。 */
    var onDismiss: (() -> Unit)? = null

    private var readingText: String = ""
    private val buffer = StringBuilder()

    private val readingView = TextView(context).apply {
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
        setPadding(dp(4f), 0, 0, 0)
    }
    private val surfaceView = TextView(context).apply {
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 22f)
        setPadding(dp(4f), dp(2f), 0, 0)
    }

    init {
        orientation = VERTICAL
        setPadding(dp(8f), dp(6f), dp(8f), dp(6f))
        setBackgroundColor(KeyboardColors.stripBackground)

        addView(readingView, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT))
        addView(surfaceView, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT))

        addView(
            LinearLayout(context).apply {
                orientation = HORIZONTAL
                gravity = Gravity.END
                setPadding(0, dp(4f), 0, 0)
                addView(chip("カタカナ") { setSurface(hira2kata(readingText)) })
                addView(chip("ひらがな") { setSurface(readingText) })
                addView(chip("登録") { onCommit?.invoke(buffer.toString()) })
                addView(chip("✕") { onDismiss?.invoke() })
            },
            LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT),
        )
        visibility = GONE
    }

    private fun chip(label: String, onClick: () -> Unit): View = TextView(context).apply {
        text = label
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 14f)
        gravity = Gravity.CENTER
        setPadding(dp(12f), dp(6f), dp(12f), dp(6f))
        setTextColor(KeyboardColors.keyText)
        background = GradientDrawable().apply {
            cornerRadius = dp(10f).toFloat()
            setColor(KeyboardColors.functionKey)
        }
        isClickable = true
        setOnClickListener { onClick() }
        layoutParams = LayoutParams(LayoutParams.WRAP_CONTENT, LayoutParams.WRAP_CONTENT)
            .apply { marginStart = dp(6f) }
    }

    fun show(reading: String, initialSurface: String) {
        readingText = reading
        buffer.setLength(0)
        buffer.append(initialSurface)
        readingView.text = "読み  $reading"
        refresh()
        visibility = VISIBLE
    }

    fun hide() {
        visibility = GONE
    }

    val surface: String get() = buffer.toString()

    fun append(text: String) {
        buffer.append(text)
        refresh()
    }

    fun deleteLast() {
        if (buffer.isNotEmpty()) {
            buffer.deleteCharAt(buffer.length - 1)
            refresh()
        }
    }

    private fun setSurface(text: String) {
        buffer.setLength(0)
        buffer.append(text)
        refresh()
    }

    private fun refresh() {
        surfaceView.text = if (buffer.isEmpty()) "|" else "$buffer|"
    }

    fun applyTheme() {
        setBackgroundColor(KeyboardColors.stripBackground)
        readingView.setTextColor(KeyboardColors.keySubText)
        surfaceView.setTextColor(KeyboardColors.keyText)
    }

    private fun hira2kata(s: String): String = buildString {
        for (ch in s) {
            val o = ch.code
            append(if (o in 0x3041..0x3096) (o + 0x60).toChar() else ch)
        }
    }

    private fun dp(value: Float): Int =
        (value * resources.displayMetrics.density).toInt()
}
