package com.nayuta.ime.keyboard

import android.content.Context
import android.graphics.drawable.GradientDrawable
import android.util.AttributeSet
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.widget.HorizontalScrollView
import android.widget.LinearLayout
import android.widget.TextView

/**
 * 変換候補の横スクロールバー。
 *
 * 候補 1 件 = 1 つの TextView を横に並べるだけの素直な作り。
 * タップで [onCandidateSelected] を呼び、確定はサービス側が行う。
 * 候補が空のときは高さを食わないよう GONE になる。
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
        setPadding(dp(4f).toInt(), 0, dp(4f).toInt(), 0)
    }

    private val candidateViews = mutableListOf<TextView>()

    init {
        isHorizontalScrollBarEnabled = false
        overScrollMode = OVER_SCROLL_NEVER
        setBackgroundColor(KeyboardColors.candidateBackground)
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

    private fun buildItem(text: String, index: Int, selected: Boolean): TextView =
        TextView(context).apply {
            this.text = text
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 18f)
            gravity = Gravity.CENTER
            setPadding(dp(18f).toInt(), 0, dp(18f).toInt(), 0)
            minWidth = dp(56f).toInt()
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.MATCH_PARENT,
            ).apply {
                marginEnd = dp(4f).toInt()
            }
            isClickable = true
            applySelection(this, selected)
            setOnClickListener {
                onCandidateSelected?.invoke(index, text)
            }
        }

    private fun applySelection(view: TextView, selected: Boolean) {
        if (selected) {
            view.setTextColor(KeyboardColors.candidateSelectedText)
            view.background = GradientDrawable().apply {
                cornerRadius = dp(10f)
                setColor(KeyboardColors.candidateSelectedBackground)
            }
        } else {
            view.setTextColor(KeyboardColors.candidateText)
            view.background = null
        }
    }

    private fun dp(value: Float): Float = TypedValue.applyDimension(
        TypedValue.COMPLEX_UNIT_DIP,
        value,
        resources.displayMetrics,
    )
}
