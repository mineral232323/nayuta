package com.nayuta.ime.keyboard

import android.content.Context
import android.content.res.ColorStateList
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.graphics.drawable.RippleDrawable
import android.util.AttributeSet
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.widget.HorizontalScrollView
import android.widget.LinearLayout
import android.widget.TextView

/**
 * 変換候補の横スクロールストリップ（Gboard のサジェスチョン行に相当）。
 *
 * Material 3 のレイアウト:
 * - 地は [KeyboardColors.stripBackground]（キーより一段明るい surface 面）
 * - 候補は 16sp のラベル。選択中だけ primaryContainer 色のピルで強調
 * - 非選択候補はタップ時にリップル（状態レイヤー）が出る
 * - 下端に hairline を 1 本入れてキーボード面と区切る
 *
 * 候補 1 件 = 1 つの TextView。タップで [onCandidateSelected] を呼び、確定は
 * サービス側が行う。候補が空のときは高さを食わないよう GONE になる。
 */
class CandidateView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : HorizontalScrollView(context, attrs) {

    /** (候補インデックス, 候補文字列) */
    var onCandidateSelected: ((Int, String) -> Unit)? = null

    private val container = LinearLayout(context).apply {
        orientation = LinearLayout.HORIZONTAL
        gravity = Gravity.CENTER_VERTICAL
        setPadding(dp(6f).toInt(), 0, dp(6f).toInt(), 0)
    }

    private val candidateViews = mutableListOf<TextView>()

    private val sans = Typeface.create("sans-serif", Typeface.NORMAL)
    private val sansMedium = Typeface.create("sans-serif-medium", Typeface.NORMAL)

    private val dividerPaint = Paint().apply { color = KeyboardColors.divider }

    init {
        isHorizontalScrollBarEnabled = false
        overScrollMode = OVER_SCROLL_NEVER
        isFillViewport = false
        setBackgroundColor(KeyboardColors.stripBackground)
        addView(container, LayoutParams(LayoutParams.WRAP_CONTENT, LayoutParams.MATCH_PARENT))
        visibility = View.GONE
    }

    /** 候補を差し替える。[selectedIndex] の候補を選択状態で描く。 */
    fun submit(candidates: List<String>, selectedIndex: Int) {
        candidateViews.clear()
        container.removeAllViews()
        if (candidates.isEmpty()) {
            visibility = View.GONE
            return
        }
        visibility = View.VISIBLE
        candidates.forEachIndexed { index, text ->
            val item = buildItem(text, index, index == selectedIndex)
            candidateViews.add(item)
            container.addView(item)
        }
        if (selectedIndex == 0) {
            post { scrollTo(0, 0) }
        }
    }

    /** 候補列はそのままに、選択位置だけ動かす（変換キーでの巡回用）。 */
    fun updateSelection(selectedIndex: Int) {
        candidateViews.forEachIndexed { index, view ->
            applySelection(view, index == selectedIndex)
        }
    }

    fun clear() = submit(emptyList(), 0)

    override fun dispatchDraw(canvas: Canvas) {
        super.dispatchDraw(canvas)
        // キーボード面との境界に hairline を 1 本
        val hairline = dp(1f)
        canvas.drawRect(0f, height - hairline, width.toFloat(), height.toFloat(), dividerPaint)
    }

    private fun buildItem(text: String, index: Int, selected: Boolean): TextView =
        TextView(context).apply {
            this.text = text
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 16f)
            gravity = Gravity.CENTER
            setPadding(dp(16f).toInt(), 0, dp(16f).toInt(), 0)
            minWidth = dp(56f).toInt()
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                dp(40f).toInt(),
            ).apply {
                marginEnd = dp(2f).toInt()
            }
            isClickable = true
            applySelection(this, selected)
            setOnClickListener {
                onCandidateSelected?.invoke(index, text)
            }
        }

    private fun applySelection(view: TextView, selected: Boolean) {
        if (selected) {
            // primaryContainer のピル + onPrimaryContainer の文字
            view.setTextColor(KeyboardColors.candidateSelectedText)
            view.typeface = sansMedium
            view.background = pill(KeyboardColors.candidateSelectedBackground)
        } else {
            view.setTextColor(KeyboardColors.candidateText)
            view.typeface = sans
            // 状態レイヤー（リップル）だけを持つ背景。ピル形状に切り抜く。
            view.background = RippleDrawable(
                ColorStateList.valueOf(KeyboardColors.ripple),
                null,
                pill(Color.WHITE),
            )
        }
    }

    private fun pill(color: Int): GradientDrawable = GradientDrawable().apply {
        cornerRadius = dp(20f)
        setColor(color)
    }

    private fun dp(value: Float): Float = TypedValue.applyDimension(
        TypedValue.COMPLEX_UNIT_DIP,
        value,
        resources.displayMetrics,
    )
}
