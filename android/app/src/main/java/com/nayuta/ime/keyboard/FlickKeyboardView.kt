package com.nayuta.ime.keyboard

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import android.util.AttributeSet
import android.util.TypedValue
import android.view.HapticFeedbackConstants
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
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
 * 見た目は Gboard（Material 3）に寄せている:
 * - 白い角丸キー + ライトグレーの地、機能キーは一段暗いトーンで区別
 * - 押下は primary の状態レイヤー（薄いブルー）、フリック方向が決まったら
 *   そのサブラベルを primary 色・少し大きめ・太字で強調
 * - 押しているキーの上に、選択中の文字を出した白い角丸ポップアップ（影 + 輪郭）
 * - 機能キーはポップアップを出さない（Gboard と同じ）
 * - フリック判定は移動距離 [flickThresholdPx] を超えてから、上下左右のうち
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
        fun onCursorMove(left: Boolean)
        fun onEnter()
        fun onTransformPrevious()
        fun onHandakutenPrevious()
        fun onNumberToggle()
        fun onKanaToggle()
        fun onCorrect()
        fun onShift()
    }

    var listener: Listener? = null

    private var layout: KeyboardLayout = KeyboardLayouts.forMode(KeyboardMode.KANA)

    /** rows × cols の矩形。layout かサイズが変わったときだけ作り直す。 */
    private val keyRects = mutableListOf<MutableList<RectF>>()

    // ---- Material 3 のタイポグラフィ ----
    private val sans = Typeface.create("sans-serif", Typeface.NORMAL)
    private val sansMedium = Typeface.create("sans-serif-medium", Typeface.NORMAL)

    private val keyPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val shadowPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val outlinePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE }
    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textAlign = Paint.Align.CENTER
        typeface = sans
    }
    private val subLabelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textAlign = Paint.Align.CENTER
        typeface = sans
    }
    private val functionLabelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textAlign = Paint.Align.CENTER
        typeface = sansMedium
    }
    private val balloonTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textAlign = Paint.Align.CENTER
        typeface = sansMedium
    }

    // ---- 押下状態 ----
    private var pressedRow = -1
    private var pressedCol = -1
    private var pressedDirection = FlickDirection.CENTER
    private var downX = 0f
    private var downY = 0f
    private var activePointerId = MotionEvent.INVALID_POINTER_ID

    /** 押下開始時刻（長押し判定用）。 */
    private var downTime = 0L

    private val repeatHandler = Handler(Looper.getMainLooper())
    private var repeatStarted = false

    /** 削除キーを押しっぱなしにしたときの連続削除。 */
    private val repeatDelete = object : Runnable {
        override fun run() {
            repeatStarted = true
            listener?.onDelete()
            repeatHandler.postDelayed(this, REPEAT_INTERVAL_MS)
        }
    }

    /** 直前のタップ位置と時刻（ダブルタップ判定用）。 */
    private var lastTapRow = -1
    private var lastTapCol = -1
    private var lastTapTime = 0L

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
        var total = edgePadding * 2
        rows.forEach { total += rowUnit * it.heightWeight }
        total += rowGap * (rows.size - 1)
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
        val totalWeight = rows.sumOf { it.heightWeight.toDouble() }.toFloat().coerceAtLeast(0.01f)
        val usableHeight = height - 2 * edgePadding - rowGap * (rows.size - 1)
        val unit = usableHeight / totalWeight
        var y = edgePadding
        rows.forEach { row ->
            val columns = row.keys.size.coerceAtLeast(1)
            val rowHeight = unit * row.heightWeight
            val usableWidth = width - 2 * edgePadding - rowGap * (columns - 1)
            val weightSum = row.keys.sumOf { it.widthWeight.toDouble() }.toFloat()
                .coerceAtLeast(0.01f)
            val unitWidth = usableWidth / weightSum
            val rects = mutableListOf<RectF>()
            var x = edgePadding
            row.keys.forEach { key ->
                val cellWidth = unitWidth * key.widthWeight
                rects.add(RectF(x, y, x + cellWidth, y + rowHeight))
                x += cellWidth + rowGap
            }
            keyRects.add(rects)
            y += rowHeight + rowGap
        }
    }

    // ------------------------------------------------------------------
    // 描画
    // ------------------------------------------------------------------

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (keyRects.isEmpty()) rebuildRects(width, height)
        layout.rows.forEachIndexed { rowIndex, row ->
            row.keys.forEachIndexed { columnIndex, key ->
                if (key is SpacerKey) return@forEachIndexed
                val rect = keyRects.getOrNull(rowIndex)?.getOrNull(columnIndex) ?: return@forEachIndexed
                val isPressedKey = rowIndex == pressedRow && columnIndex == pressedCol
                drawKeyBackground(canvas, rect, key, isPressedKey)
                when (key) {
                    is FlickKey -> drawFlickKey(canvas, rect, key, isPressedKey)
                    is FunctionKey -> drawFunctionKey(canvas, rect, key)
                    is SpacerKey -> Unit
                }
            }
        }
        drawBalloonIfNeeded(canvas)
    }

    private fun drawKeyBackground(canvas: Canvas, rect: RectF, key: KeySpec, isPressedKey: Boolean) {
        keyPaint.color = when {
            key is FunctionKey && isPressedKey -> KeyboardColors.functionKeyPressed
            key is FunctionKey -> KeyboardColors.functionKey
            isPressedKey -> KeyboardColors.keyPressed
            else -> KeyboardColors.key
        }
        canvas.drawRoundRect(rect, keyCornerRadius, keyCornerRadius, keyPaint)
    }

    private fun drawFlickKey(canvas: Canvas, rect: RectF, key: FlickKey, isPressedKey: Boolean) {
        val centerX = rect.centerX()
        val centerY = rect.centerY()

        labelPaint.textSize = sp(20f)
        labelPaint.color = KeyboardColors.keyText
        canvas.drawText(key.label, centerX, baselineFor(labelPaint, centerY), labelPaint)

        key.subLabels.forEach { (direction, text) ->
            val active = isPressedKey && direction == pressedDirection
            subLabelPaint.textSize = sp(if (active) 13f else 11f)
            subLabelPaint.color = if (active) {
                KeyboardColors.keySubTextActive
            } else {
                KeyboardColors.keySubText
            }
            subLabelPaint.typeface = if (active) sansMedium else sans
            val (x, y) = subLabelCenter(rect, direction)
            canvas.drawText(text, x, baselineFor(subLabelPaint, y), subLabelPaint)
        }
    }

    private fun drawFunctionKey(canvas: Canvas, rect: RectF, key: FunctionKey) {
        functionLabelPaint.textSize = sp(13.5f)
        functionLabelPaint.color = KeyboardColors.keyText
        canvas.drawText(
            key.label,
            rect.centerX(),
            baselineFor(functionLabelPaint, rect.centerY()),
            functionLabelPaint,
        )
    }

    /**
     * 押しているフリックキーの上に、選択中の文字を出す Gboard 風ポップアップ。
     * 影は半透明の角丸を 3 枚重ねて近似している（ハードウェア描画でも ShadowLayer に
     * 頼らず済むようにするため）。
     */
    private fun drawBalloonIfNeeded(canvas: Canvas) {
        val row = layout.rows.getOrNull(pressedRow) ?: return
        val key = row.keys.getOrNull(pressedCol) as? FlickKey ?: return
        val anchor = keyRects.getOrNull(pressedRow)?.getOrNull(pressedCol) ?: return
        val text = key.output(pressedDirection) ?: return

        balloonTextPaint.textSize = sp(21f)
        val horizontalPadding = dp(16f)
        val balloonHeight = dp(44f)
        val width = balloonTextPaint.measureText(text) + horizontalPadding * 2
        val maxLeft = (this.width - width - dp(4f)).coerceAtLeast(dp(4f))
        val left = (anchor.centerX() - width / 2).coerceIn(dp(4f), maxLeft)
        // 最上段のキーでは上に出すと画面外になるので、キーの上端付近に重ねる。
        val top = (anchor.top - balloonHeight - dp(8f)).coerceAtLeast(dp(2f))
        val bubble = RectF(left, top, left + width, top + balloonHeight)
        val radius = dp(13f)

        // 影（3 層の半透明角丸）
        for (layer in 3 downTo 1) {
            val grow = dp(1.2f) * layer
            shadowPaint.color = KeyboardColors.balloonShadow
            canvas.drawRoundRect(
                bubble.left - grow,
                bubble.top - grow + dp(1.5f),
                bubble.right + grow,
                bubble.bottom + grow + dp(1.5f),
                radius + grow,
                radius + grow,
                shadowPaint,
            )
        }

        keyPaint.color = KeyboardColors.balloonBackground
        canvas.drawRoundRect(bubble, radius, radius, keyPaint)

        outlinePaint.color = KeyboardColors.balloonOutline
        outlinePaint.strokeWidth = dp(0.7f)
        canvas.drawRoundRect(bubble, radius, radius, outlinePaint)

        balloonTextPaint.color = KeyboardColors.balloonText
        canvas.drawText(
            text,
            bubble.centerX(),
            baselineFor(balloonTextPaint, bubble.centerY()),
            balloonTextPaint,
        )
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
                downTime = event.eventTime
                // 削除キーは長押しで連続削除
                val pressedKey = layout.rows.getOrNull(hit.first)?.keys?.getOrNull(hit.second)
                if (pressedKey is FunctionKey && pressedKey.action == KeyAction.DELETE) {
                    repeatStarted = false
                    repeatHandler.postDelayed(repeatDelete, LONG_PRESS_MS)
                }
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
                val longPress = event.eventTime - downTime > LONG_PRESS_MS
                repeatHandler.removeCallbacks(repeatDelete)
                val repeated = repeatStarted
                repeatStarted = false
                clearPressed()
                if (repeated) {
                    return true
                }
                if (longPress) {
                    emitLongPress(rowIndex, columnIndex)
                } else {
                    emit(rowIndex, columnIndex, direction)
                }
                performClick()
                return true
            }

            MotionEvent.ACTION_CANCEL -> {
                repeatHandler.removeCallbacks(repeatDelete)
                repeatStarted = false
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
                    // スペーサーは押せない
                    val key = layout.rows.getOrNull(rowIndex)?.keys?.getOrNull(columnIndex)
                    if (key is SpacerKey) return null
                    return rowIndex to columnIndex
                }
            }
        }
        return null
    }

    /** 長押し時の動作。未定義なら通常の入力をそのまま行う。 */
    private fun emitLongPress(rowIndex: Int, columnIndex: Int) {
        val key = layout.rows.getOrNull(rowIndex)?.keys?.getOrNull(columnIndex)
        val action = (key as? FunctionKey)?.longPressAction
        if (action != null) {
            dispatch(action)
        } else {
            emit(rowIndex, columnIndex, FlickDirection.CENTER)
        }
    }

    private fun dispatch(action: KeyAction) {
        when (action) {
            KeyAction.DELETE -> listener?.onDelete()
            KeyAction.CONVERT -> listener?.onConvert()
            KeyAction.SPACE -> listener?.onSpace()
            KeyAction.MODE_TOGGLE -> listener?.onModeToggle()
            KeyAction.CURSOR_LEFT -> listener?.onCursorMove(left = true)
            KeyAction.CURSOR_RIGHT -> listener?.onCursorMove(left = false)
            KeyAction.ENTER -> listener?.onEnter()
            KeyAction.TRANSFORM -> listener?.onTransformPrevious()
            KeyAction.HANDAKUTEN -> listener?.onHandakutenPrevious()
            KeyAction.NUMBER_TOGGLE -> listener?.onNumberToggle()
            KeyAction.KANA_TOGGLE -> listener?.onKanaToggle()
            KeyAction.CORRECT -> listener?.onCorrect()
            KeyAction.SHIFT -> listener?.onShift()
        }
    }

    private fun emit(rowIndex: Int, columnIndex: Int, direction: FlickDirection) {
        val key = layout.rows.getOrNull(rowIndex)?.keys?.getOrNull(columnIndex) ?: return
        // ダブルタップで別動作になるキー（小゛ → 半濁点など）
        if (key is FunctionKey && key.doubleTapAction != null && direction == FlickDirection.CENTER) {
            val now = SystemClock.uptimeMillis()
            val isDouble = rowIndex == lastTapRow &&
                columnIndex == lastTapCol &&
                now - lastTapTime <= DOUBLE_TAP_MS
            lastTapRow = rowIndex
            lastTapCol = columnIndex
            lastTapTime = if (isDouble) 0L else now
            if (isDouble) {
                dispatch(key.doubleTapAction!!)
                return
            }
        }
        when (key) {
            is FlickKey -> key.output(direction)?.let { listener?.onText(it) }
            is FunctionKey -> dispatch(key.action)
            is SpacerKey -> Unit
        }
    }

    // ------------------------------------------------------------------
    // 寸法・単位変換
    // ------------------------------------------------------------------

    /** 1 行ぶんの基準高さ（heightWeight = 1 のとき）。 */
    private val rowUnit: Float get() = dp(50f)

    /** キー間のすき間。Material の 4dp グリッド。 */
    private val rowGap: Float get() = dp(4f)

    /** キーボード外周の余白。 */
    private val edgePadding: Float get() = dp(4f)

    /** キーの角丸。Material 3 の medium shape（12dp）に近い値。 */
    private val keyCornerRadius: Float get() = dp(10f)

    /** ダーク/ライト切替の再適用（ラベルの色は描画時に参照しているので背景だけでよい）。 */
    fun applyTheme() {
        setBackgroundColor(KeyboardColors.keyboardBackground)
        invalidate()
    }

    override fun onDetachedFromWindow() {
        // 連続削除が残らないようにする
        repeatHandler.removeCallbacks(repeatDelete)
        repeatStarted = false
        super.onDetachedFromWindow()
    }

    private companion object {
        /** 長押しとみなす時間。 */
        const val LONG_PRESS_MS = 400L

        /** ダブルタップとみなす間隔。 */
        const val DOUBLE_TAP_MS = 300L

        /** 長押しで連続削除するときの間隔。 */
        const val REPEAT_INTERVAL_MS = 60L
    }

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
