package com.nayuta.ime.keyboard

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import android.util.AttributeSet
import android.util.TypedValue
import android.view.HapticFeedbackConstants
import android.view.MotionEvent
import android.view.View
import kotlin.math.abs
import kotlin.math.hypot

/**
 * 12 キーフリック配列を自前で描画・入力処理する View。
 *
 * 配列の知識は一切持たず、[KeyboardLayouts] が渡す [KeyboardLayout] を
 * そのまま矩形に並べて描くだけにしてある（配列の変更はテーブル側だけで完結する）。
 *
 * - 押下中はキーをハイライトし、フリック方向が確定したらサブかなを強調する
 * - 押しているキーの上に、選択中の文字を出した吹き出し（ポップアップ）を描く
 * - フリック判定は移動距離 [flickThresholdDp] を超えてから、上下左右のうち
 *   移動量が大きい軸で決める（8 方向ではなく 4 方向）
 */
class FlickKeyboardView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {

    /** サービス側への入力通知。キーボードは「どの文字か」を知るだけで、変換は知らない。 */
    interface Listener {
        fun onText(text: String)
        fun onDelete()
        fun onConvert()
        fun onSpace()
        fun onModeToggle()
    }

    var listener: Listener? = null

    private var layout: KeyboardLayout = KeyboardLayouts.forMode(KeyboardMode.KANA)

    /** rows × cols の矩形。layout かサイズが変わったときだけ作り直す。 */
    private val keyRects = mutableListOf<MutableList<RectF>>()

    private val keyPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textAlign = Paint.Align.CENTER
        typeface = Typeface.DEFAULT
    }
    private val subLabelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textAlign = Paint.Align.CENTER
        typeface = Typeface.DEFAULT
    }
    private val functionLabelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textAlign = Paint.Align.CENTER
        typeface = Typeface.DEFAULT_BOLD
    }
    private val balloonPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textAlign = Paint.Align.CENTER
        typeface = Typeface.DEFAULT_BOLD
    }

    // ---- 押下状態 ----
    private var pressedRow = -1
    private var pressedCol = -1
    private var pressedDirection = FlickDirection.CENTER
    private var downX = 0f
    private var downY = 0f
    private var activePointerId = MotionEvent.INVALID_POINTER_ID

    private val flickThresholdPx: Float get() = dp(18f)

    init {
        setBackgroundColor(KeyboardColors.keyboardBackground)
    }

    // ------------------------------------------------------------------
    // 公開 API
    // ------------------------------------------------------------------

    fun setLayout(layout: KeyboardLayout) {
        this.layout = layout
        clearPressed()
        keyRects.clear()
        requestLayout()
        invalidate()
    }

    // ------------------------------------------------------------------
    // 計測・レイアウト
    // ------------------------------------------------------------------

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        val width = MeasureSpec.getSize(widthMeasureSpec)
        setMeasuredDimension(width, desiredHeight().toInt())
    }

    private fun desiredHeight(): Float {
        val rows = layout.rows
        if (rows.isEmpty()) return dp(240f)
        var total = dp(5f) * 2
        rows.forEach { total += dp(52f) * it.heightWeight }
        total += dp(5f) * (rows.size - 1)
        return total
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        rebuildRects(w, h)
    }

    private fun rebuildRects(width: Int, height: Int) {
        keyRects.clear()
        if (width <= 0 || height <= 0) return
        val rows = layout.rows
        if (rows.isEmpty()) return
        val hPad = dp(5f)
        val vPad = dp(5f)
        val gap = dp(5f)
        val totalWeight = rows.sumOf { it.heightWeight.toDouble() }.toFloat().coerceAtLeast(0.01f)
        val usableHeight = height - 2 * vPad - gap * (rows.size - 1)
        val unit = usableHeight / totalWeight
        var y = vPad
        rows.forEach { row ->
            val columns = row.keys.size.coerceAtLeast(1)
            val rowHeight = unit * row.heightWeight
            val usableWidth = width - 2 * hPad - gap * (columns - 1)
            val cellWidth = usableWidth / columns
            val rects = mutableListOf<RectF>()
            var x = hPad
            repeat(columns) {
                rects.add(RectF(x, y, x + cellWidth, y + rowHeight))
                x += cellWidth + gap
            }
            keyRects.add(rects)
            y += rowHeight + gap
        }
    }

    // ------------------------------------------------------------------
    // 描画
    // ------------------------------------------------------------------

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (keyRects.isEmpty()) rebuildRects(width, height)
        val corner = dp(7f)
        layout.rows.forEachIndexed { rowIndex, row ->
            row.keys.forEachIndexed { columnIndex, key ->
                val rect = keyRects.getOrNull(rowIndex)?.getOrNull(columnIndex) ?: return@forEachIndexed
                val isPressedKey = rowIndex == pressedRow && columnIndex == pressedCol
                keyPaint.color = if (isPressedKey) KeyboardColors.keyPressed else KeyboardColors.key
                canvas.drawRoundRect(rect, corner, corner, keyPaint)
                when (key) {
                    is FlickKey -> drawFlickKey(canvas, rect, key, isPressedKey)
                    is FunctionKey -> drawFunctionKey(canvas, rect, key)
                }
            }
        }
        drawBalloonIfNeeded(canvas)
    }

    private fun drawFlickKey(canvas: Canvas, rect: RectF, key: FlickKey, isPressedKey: Boolean) {
        val centerX = rect.centerX()
        val centerY = rect.centerY()

        labelPaint.textSize = sp(19f)
        labelPaint.color = KeyboardColors.keyText
        canvas.drawText(key.label, centerX, baselineFor(labelPaint, centerY), labelPaint)

        subLabelPaint.textSize = sp(11f)
        key.subLabels.forEach { (direction, text) ->
            val active = isPressedKey && direction == pressedDirection
            subLabelPaint.color = if (active) {
                KeyboardColors.keySubTextActive
            } else {
                KeyboardColors.keySubText
            }
            val position = subLabelCenter(rect, direction)
            canvas.drawText(text, position.first, baselineFor(subLabelPaint, position.second), subLabelPaint)
        }
    }

    private fun drawFunctionKey(canvas: Canvas, rect: RectF, key: FunctionKey) {
        functionLabelPaint.textSize = sp(15f)
        functionLabelPaint.color = KeyboardColors.keyText
        canvas.drawText(
            key.label,
            rect.centerX(),
            baselineFor(functionLabelPaint, rect.centerY()),
            functionLabelPaint,
        )
    }

    private fun drawBalloonIfNeeded(canvas: Canvas) {
        val row = layout.rows.getOrNull(pressedRow) ?: return
        val key = row.keys.getOrNull(pressedCol) ?: return
        val anchor = keyRects.getOrNull(pressedRow)?.getOrNull(pressedCol) ?: return
        val text = balloonTextFor(key) ?: return

        balloonPaint.textSize = sp(18f)
        val horizontalPadding = dp(13f)
        val balloonHeight = dp(38f)
        val width = balloonPaint.measureText(text) + horizontalPadding * 2
        val maxLeft = (this.width - width - dp(4f)).coerceAtLeast(dp(4f))
        val left = (anchor.centerX() - width / 2).coerceIn(dp(4f), maxLeft)
        // 最上段のキーでは上に出すと画面外になるので、キーの上端に重ねて表示する。
        val above = anchor.top - balloonHeight - dp(8f)
        val top = if (above >= 0f) above else anchor.top + dp(6f)
        val bubble = RectF(left, top, left + width, top + balloonHeight)

        keyPaint.color = KeyboardColors.balloonBackground
        canvas.drawRoundRect(bubble, dp(9f), dp(9f), keyPaint)
        balloonPaint.color = KeyboardColors.balloonText
        canvas.drawText(
            text,
            bubble.centerX(),
            baselineFor(balloonPaint, bubble.centerY()),
            balloonPaint,
        )
    }

    private fun balloonTextFor(key: KeySpec): String? = when (key) {
        is FlickKey -> key.output(pressedDirection) ?: key.label
        is FunctionKey -> key.label
    }

    private fun subLabelCenter(rect: RectF, direction: FlickDirection): Pair<Float, Float> = when (direction) {
        FlickDirection.UP -> rect.centerX() to rect.top + dp(10f)
        FlickDirection.DOWN -> rect.centerX() to rect.bottom - dp(10f)
        FlickDirection.LEFT -> rect.left + dp(13f) to rect.centerY()
        FlickDirection.RIGHT -> rect.right - dp(13f) to rect.centerY()
        FlickDirection.CENTER -> rect.centerX() to rect.centerY()
    }

    // ------------------------------------------------------------------
    // タッチ入力
    // ------------------------------------------------------------------

    override fun onTouchEvent(event: MotionEvent): Boolean {
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                val hit = hitTest(event.x, event.y) ?: return false
                pressedRow = hit.first
                pressedCol = hit.second
                pressedDirection = FlickDirection.CENTER
                downX = event.x
                downY = event.y
                activePointerId = event.getPointerId(0)
                isPressed = true
                performHapticFeedback(HapticFeedbackConstants.VIRTUAL_KEY)
                invalidate()
                return true
            }

            MotionEvent.ACTION_MOVE -> {
                if (pressedRow < 0) return false
                val index = event.findPointerIndex(activePointerId)
                if (index < 0) return false
                val direction = directionFor(
                    event.getX(index) - downX,
                    event.getY(index) - downY,
                )
                if (direction != pressedDirection) {
                    pressedDirection = direction
                    performHapticFeedback(HapticFeedbackConstants.KEYBOARD_TAP)
                    invalidate()
                }
                return true
            }

            MotionEvent.ACTION_UP -> {
                if (pressedRow < 0) return false
                val rowIndex = pressedRow
                val columnIndex = pressedCol
                val direction = pressedDirection
                clearPressed()
                emit(rowIndex, columnIndex, direction)
                performClick()
                return true
            }

            MotionEvent.ACTION_CANCEL -> {
                clearPressed()
                return true
            }
        }
        return super.onTouchEvent(event)
    }

    override fun performClick(): Boolean {
        super.performClick()
        return true
    }

    private fun clearPressed() {
        if (pressedRow < 0 && pressedCol < 0) return
        pressedRow = -1
        pressedCol = -1
        pressedDirection = FlickDirection.CENTER
        activePointerId = MotionEvent.INVALID_POINTER_ID
        isPressed = false
        invalidate()
    }

    /** 指の移動量からフリック方向を決める。閾値未満はタップ扱い。 */
    private fun directionFor(dx: Float, dy: Float): FlickDirection {
        if (hypot(dx, dy) < flickThresholdPx) return FlickDirection.CENTER
        return if (abs(dx) > abs(dy)) {
            if (dx > 0) FlickDirection.RIGHT else FlickDirection.LEFT
        } else {
            if (dy > 0) FlickDirection.DOWN else FlickDirection.UP
        }
    }

    private fun hitTest(x: Float, y: Float): Pair<Int, Int>? {
        // キー間の隙間も押せるように、矩形を少し広げて判定する（タップの取りこぼし防止）。
        val slop = dp(3f)
        keyRects.forEachIndexed { rowIndex, rects ->
            rects.forEachIndexed { columnIndex, rect ->
                if (x >= rect.left - slop && x <= rect.right + slop &&
                    y >= rect.top - slop && y <= rect.bottom + slop
                ) {
                    return rowIndex to columnIndex
                }
            }
        }
        return null
    }

    private fun emit(rowIndex: Int, columnIndex: Int, direction: FlickDirection) {
        val key = layout.rows.getOrNull(rowIndex)?.keys?.getOrNull(columnIndex) ?: return
        when (key) {
            is FlickKey -> key.output(direction)?.let { listener?.onText(it) }
            is FunctionKey -> when (key.action) {
                KeyAction.DELETE -> listener?.onDelete()
                KeyAction.CONVERT -> listener?.onConvert()
                KeyAction.SPACE -> listener?.onSpace()
                KeyAction.MODE_TOGGLE -> listener?.onModeToggle()
            }
        }
    }

    // ------------------------------------------------------------------
    // 単位変換ヘルパ
    // ------------------------------------------------------------------

    private fun dp(value: Float): Float = TypedValue.applyDimension(
        TypedValue.COMPLEX_UNIT_DIP,
        value,
        resources.displayMetrics,
    )

    private fun sp(value: Float): Float = TypedValue.applyDimension(
        TypedValue.COMPLEX_UNIT_SP,
        value,
        resources.displayMetrics,
    )

    /** テキストの縦中心を centerY に合わせるためのベースライン。 */
    private fun baselineFor(paint: Paint, centerY: Float): Float =
        centerY - (paint.ascent() + paint.descent()) / 2f
}
