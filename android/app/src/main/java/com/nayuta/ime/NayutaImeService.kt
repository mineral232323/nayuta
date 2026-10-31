package com.nayuta.ime

import android.inputmethodservice.InputMethodService
import android.os.Handler
import android.os.Looper
import android.view.KeyEvent
import android.view.View
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputConnection
import android.widget.LinearLayout
import com.nayuta.ime.converter.Converter
import com.nayuta.ime.converter.Converters
import com.nayuta.ime.converter.dict.UserDictionary
import com.nayuta.ime.keyboard.CandidateView
import com.nayuta.ime.keyboard.CorrectionPanel
import com.nayuta.ime.keyboard.FlickKeyboardView
import com.nayuta.ime.keyboard.KeyboardColors
import com.nayuta.ime.keyboard.KeyboardLayouts
import com.nayuta.ime.keyboard.KeyboardMode
import com.nayuta.ime.keyboard.KanaMarks
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/** 変換中の状態。 */
private enum class ComposingState {
    /** 何も入力していない。 */
    IDLE,

    /** 読みを入力中。 */
    COMPOSING,

    /** 候補を表示中（変換キー / 空白キーで入る）。 */
    CONVERTING,
}

/**
 * Nayuta の IME 本体。
 *
 * 入力ビューは「候補バー + 12 キーフリックキーボード」の 2 段。
 * 変換ロジックは [Converter] の向こう側だけにあり、このクラスは
 * 「読みを貯める → convert() に渡す → 候補を出す → 確定する」の状態遷移だけを持つ。
 *
 * 状態遷移:
 * ```
 * IDLE --かな入力--> COMPOSING --変換/空白--> CONVERTING --タップ/確定--> IDLE
 *   ^                    |                        |
 *   +------- 削除で空 -----+--- 削除で読みに戻る ----+
 * ```
 * 変換はメインスレッドを塞がないよう単一スレッドの executor で実行し、
 * 結果だけメインスレッドに戻す（将来の ONNX 推論を見越した作り）。
 */
class NayutaImeService : InputMethodService(), FlickKeyboardView.Listener {

    private var converter: Converter? = null
    private var convertExecutor: ExecutorService? = null
    private val uiHandler = Handler(Looper.getMainLooper())

    private var keyboardView: FlickKeyboardView? = null
    private var candidateView: CandidateView? = null

    /** 未確定の読み（ひらがな）。 */
    private val reading = StringBuilder()

    private var candidates: List<String> = emptyList()

    /**
     * ライブ変換のプレビュー。読みは [reading] に貯めたまま、表示だけ変換結果にする。
     * null のときは読み（かな）をそのまま表示する。
     */
    private var livePreview: String? = null

    /** 入力が止まってから変換を発火するまでの待ち時間。 */
    private var liveJob: Runnable? = null

    // ---- 訂正パネル ----
    private var correctionView: CorrectionPanel? = null
    private var correcting = false
    private var correctReading = ""
    private var correctingFromComposing = false

    /** 直前に確定した語（訂正の対象）。 */
    private var lastCommittedReading: String? = null
    private var lastCommittedSurface: String? = null
    private var selectedIndex = 0
    private var state: ComposingState = ComposingState.IDLE
    private var mode: KeyboardMode = KeyboardMode.KANA

    /** 変換要求の世代番号。古い非同期結果を捨てるために使う。 */
    private var convertGeneration = 0

    // ------------------------------------------------------------------
    // ライフサイクル
    // ------------------------------------------------------------------

    override fun onCreate() {
        super.onCreate()
        converter = Converters.get(this)
        convertExecutor = Executors.newSingleThreadExecutor { runnable ->
            Thread(runnable, "nayuta-convert")
        }
    }

    override fun onDestroy() {
        convertExecutor?.shutdownNow()
        convertExecutor = null
        uiHandler.removeCallbacksAndMessages(null)
        super.onDestroy()
    }

    override fun onCreateInputView(): View {
        // 端末のダークモード設定を反映してから組む
        KeyboardColors.resolve(this)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(KeyboardColors.keyboardBackground)
        }

        val correction = CorrectionPanel(this).apply {
            onCommit = { surface -> commitCorrection(surface) }
            onDismiss = { closeCorrection() }
        }
        val candidates = CandidateView(this).apply {
            onCandidateSelected = { index, _ -> selectCandidate(index) }
        }
        val keyboard = FlickKeyboardView(this).apply {
            listener = this@NayutaImeService
            setLayout(KeyboardLayouts.forMode(mode))
        }

        root.addView(
            correction,
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT,
            ),
        )
        root.addView(
            candidates,
            // Gboard のサジェスチョン行と同じ 48dp（Material の最小タップ領域）
            LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dp(48)),
        )
        root.addView(
            keyboard,
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT,
            ),
        )

        candidateView = candidates
        keyboardView = keyboard
        correctionView = correction
        return root
    }

    override fun onConfigurationChanged(newConfig: android.content.res.Configuration) {
        super.onConfigurationChanged(newConfig)
        val wasDark = KeyboardColors.isDark
        if (KeyboardColors.resolve(this) == wasDark) return
        // テーマが変わったときだけ塗り直す
        candidateView?.applyTheme()
        correctionView?.applyTheme()
        keyboardView?.applyTheme()
        (keyboardView?.parent as? View)?.setBackgroundColor(KeyboardColors.keyboardBackground)
    }

    override fun onStartInputView(info: EditorInfo?, restarting: Boolean) {
        super.onStartInputView(info, restarting)
        resetComposing(finishComposing = true)
        keyboardView?.setLayout(KeyboardLayouts.forMode(mode))
    }

    override fun onFinishInput() {
        super.onFinishInput()
        resetComposing(finishComposing = true)
    }

    /** 横画面でも全画面抽出モードにはしない（キーボードのみ表示）。 */
    override fun onEvaluateFullscreenMode(): Boolean = false

    override fun onKeyDown(keyCode: Int, event: KeyEvent): Boolean {
        // 入力中に戻るキー: まず未確定入力を消す。空なら通常どおり閉じる。
        if (keyCode == KeyEvent.KEYCODE_BACK && state != ComposingState.IDLE) {
            resetComposing(finishComposing = true)
            return true
        }
        return super.onKeyDown(keyCode, event)
    }

    // ------------------------------------------------------------------
    // キーボードからの入力
    // ------------------------------------------------------------------

    override fun onText(text: String) {
        if (correcting) {
            correctionView?.append(text)
            return
        }
        if (state == ComposingState.CONVERTING) commitCurrent()

        when (text) {
            KanaMarks.DAKUTEN -> applyMark(DAKUTEN_MAP, KanaMarks.DAKUTEN_CHAR)
            KanaMarks.HANDAKUTEN -> applyMark(HANDAKUTEN_MAP, KanaMarks.HANDAKUTEN_CHAR)
            else -> reading.append(text)
        }

        candidates = emptyList()
        selectedIndex = 0
        livePreview = null
        state = if (reading.isEmpty()) ComposingState.IDLE else ComposingState.COMPOSING
        render()
        scheduleLiveConversion()
    }

    /** 入力が 500ms 止まったら変換を走らせる（ライブ変換）。 */
    private fun scheduleLiveConversion() {
        cancelLiveConversion()
        if (reading.isEmpty()) return
        val target = reading.toString()
        val job = Runnable { startLiveConversion(target) }
        liveJob = job
        uiHandler.postDelayed(job, LIVE_CONVERT_DELAY_MS)
    }

    private fun cancelLiveConversion() {
        liveJob?.let { uiHandler.removeCallbacks(it) }
        liveJob = null
    }

    /**
     * ライブ変換。読みは保持したまま、表示だけ変換結果に差し替える。
     * 明示的な「変換」と違い候補ストリップは出さず、確定もしない。
     */
    private fun startLiveConversion(target: String) {
        if (state != ComposingState.COMPOSING) return
        if (reading.toString() != target) return
        val engine = converter ?: return
        val executor = convertExecutor ?: return
        val generation = ++convertGeneration
        val contextBefore = contextBeforeCursor()

        executor.execute {
            val result = try {
                engine.convert(target, contextBefore)
            } catch (e: Exception) {
                android.util.Log.e("NayutaIme", "live convert failed: $target", e)
                emptyList()
            }
            uiHandler.post {
                if (generation != convertGeneration) return@post
                if (reading.toString() != target) return@post
                if (state != ComposingState.COMPOSING) return@post
                livePreview = result.firstOrNull() ?: target
                render(candidatesChanged = false)
            }
        }
    }

    override fun onDelete() {
        if (correcting) {
            correctionView?.deleteLast()
            return
        }
        when (state) {
            ComposingState.CONVERTING -> {
                // 変換中に削除 → 読みの編集に戻る
                candidates = emptyList()
                selectedIndex = 0
                livePreview = null
                state = if (reading.isEmpty()) ComposingState.IDLE else ComposingState.COMPOSING
                render()
                scheduleLiveConversion()
            }

            ComposingState.COMPOSING -> {
                if (reading.isNotEmpty()) {
                    reading.deleteCharAt(reading.length - 1)
                }
                livePreview = null
                if (reading.isEmpty()) {
                    candidates = emptyList()
                    selectedIndex = 0
                    state = ComposingState.IDLE
                    cancelLiveConversion()
                }
                render()
                scheduleLiveConversion()
            }

            ComposingState.IDLE -> deleteOneCharBeforeCursor()
        }
    }

    override fun onConvert() {
        if (correcting) return
        when (state) {
            ComposingState.CONVERTING -> {
                if (candidates.size > 1) {
                    selectedIndex = (selectedIndex + 1) % candidates.size
                    render(candidatesChanged = false)
                }
            }

            ComposingState.COMPOSING -> startConversion()
            ComposingState.IDLE -> Unit
        }
    }

    override fun onSpace() {
        when (state) {
            ComposingState.CONVERTING -> {
                commitCurrent()
                insertText(spaceForMode())
            }

            // かな入力中は空白キー = 変換（一般的な日本語 IME の挙動）
            ComposingState.COMPOSING -> startConversion()
            ComposingState.IDLE -> insertText(spaceForMode())
        }
    }

    override fun onCursorMove(left: Boolean) {
        val ic = currentInputConnection ?: return
        // 変換中なら確定してから動かす（編集中の読みを壊さない）
        if (state != ComposingState.IDLE) commitCurrent()
        val code = if (left) KeyEvent.KEYCODE_DPAD_LEFT else KeyEvent.KEYCODE_DPAD_RIGHT
        ic.sendKeyEvent(KeyEvent(KeyEvent.ACTION_DOWN, code))
        ic.sendKeyEvent(KeyEvent(KeyEvent.ACTION_UP, code))
    }

    override fun onEnter() {
        val ic = currentInputConnection ?: return
        when (state) {
            ComposingState.CONVERTING -> commitCurrent()
            ComposingState.COMPOSING -> {
                // 変換前なら確定扱い（読みをそのまま出す）
                commitCurrent()
            }
            ComposingState.IDLE -> Unit
        }
        ic.sendKeyEvent(KeyEvent(KeyEvent.ACTION_DOWN, KeyEvent.KEYCODE_ENTER))
        ic.sendKeyEvent(KeyEvent(KeyEvent.ACTION_UP, KeyEvent.KEYCODE_ENTER))
    }

    /**
     * 訂正。いま変換しようとしている語（または直前に確定した語）を
     * パネルに出し、表記を直してユーザー辞書へ登録できるようにする。
     */
    override fun onCorrect() {
        if (correcting) {
            closeCorrection()
            return
        }
        val ic = currentInputConnection ?: return

        when (state) {
            ComposingState.COMPOSING, ComposingState.CONVERTING -> {
                val target = reading.toString()
                if (target.isEmpty()) return
                val shown = livePreview ?: candidates.getOrNull(selectedIndex) ?: target
                openCorrection(target, shown, fromComposing = true)
            }

            ComposingState.IDLE -> {
                val r = lastCommittedReading
                val s = lastCommittedSurface
                if (r.isNullOrEmpty() || s.isNullOrEmpty()) {
                    toast("訂正できる語がありません")
                    return
                }
                val before = ic.getTextBeforeCursor(s.length, 0)?.toString()
                if (before != s) {
                    toast("直前に確定した語だけ訂正できます")
                    return
                }
                openCorrection(r, s, fromComposing = false)
            }
        }
    }

    private fun openCorrection(readingTarget: String, surface: String, fromComposing: Boolean) {
        cancelLiveConversion()
        correctReading = readingTarget
        correctingFromComposing = fromComposing
        correcting = true
        correctionView?.show(readingTarget, surface)
        candidateView?.clear()
    }

    private fun closeCorrection() {
        correcting = false
        correctionView?.hide()
    }

    /** パネルで「登録」が押された。辞書に入れて、アプリ側のテキストも置き換える。 */
    private fun commitCorrection(surface: String) {
        if (surface.isEmpty()) return
        val ic = currentInputConnection
        UserDictionary.get(applicationContext).add(correctReading, surface)

        if (correctingFromComposing) {
            ic?.commitText(surface, 1)
            resetComposing(finishComposing = true)
        } else {
            val old = lastCommittedSurface
            if (old != null && old.isNotEmpty()) ic?.deleteSurroundingText(old.length, 0)
            ic?.commitText(surface, 1)
            lastCommittedSurface = surface
        }
        lastCommittedReading = null
        closeCorrection()
    }

    private fun toast(message: String) {
        android.widget.Toast.makeText(this, message, android.widget.Toast.LENGTH_SHORT).show()
    }

    override fun onNumberToggle() {
        if (state != ComposingState.IDLE) commitCurrent()
        mode = if (mode == KeyboardMode.NUMBER) KeyboardMode.KANA else KeyboardMode.NUMBER
        keyboardView?.setLayout(KeyboardLayouts.forMode(mode))
    }

    override fun onKanaToggle() {
        if (state != ComposingState.IDLE) commitCurrent()
        mode = KeyboardMode.KANA
        keyboardView?.setLayout(KeyboardLayouts.forMode(mode))
    }

    override fun onModeToggle() {
        if (state != ComposingState.IDLE) commitCurrent()
        mode = mode.toggled()
        keyboardView?.setLayout(KeyboardLayouts.forMode(mode))
    }

    // ------------------------------------------------------------------
    // 変換
    // ------------------------------------------------------------------

    /**
     * 変換を開始する。実エンジン（単語ラティス + BERT）は重いので
     * 別スレッドに投げ、戻ってきたら世代番号で古い結果を捨てる。
     */
    private fun startConversion() {
        cancelLiveConversion()
        livePreview = null
        val target = reading.toString()
        if (target.isEmpty()) return

        val executor = convertExecutor ?: return
        val engine = converter ?: return
        val generation = ++convertGeneration
        val contextBefore = contextBeforeCursor()

        executor.execute {
            val result = try {
                engine.convert(target, contextBefore)
            } catch (e: Exception) {
                android.util.Log.e("NayutaIme", "convert() failed: $target", e)
                emptyList()
            }
            uiHandler.post {
                if (generation != convertGeneration) return@post
                if (reading.toString() != target) return@post
                candidates = if (result.isEmpty()) listOf(target) else result
                selectedIndex = 0
                state = ComposingState.CONVERTING
                render()
            }
        }
    }

    private fun selectCandidate(index: Int) {
        if (state != ComposingState.CONVERTING) return
        if (candidates.isEmpty()) return
        selectedIndex = index.coerceIn(0, candidates.size - 1)
        commitCurrent()
    }

    /** 選択中の候補（未変換なら読み）を確定する。 */
    private fun commitCurrent() {
        val ic = currentInputConnection
        val text = candidates.getOrNull(selectedIndex) ?: livePreview ?: reading.toString()
        if (ic != null && text.isNotEmpty()) {
            ic.commitText(text, 1)
            if (reading.isNotEmpty()) {
                lastCommittedReading = reading.toString()
                lastCommittedSurface = text
            }
        }
        resetComposing(finishComposing = ic == null)
    }

    /** 変換対象より前の文脈。COMPSING の読みは文脈から除く。 */
    private fun contextBeforeCursor(maxLength: Int = CONTEXT_MAX_LENGTH): String {
        val ic = currentInputConnection ?: return ""
        val readingText = livePreview ?: reading.toString()
        val before = ic.getTextBeforeCursor(maxLength + readingText.length, 0)?.toString() ?: return ""
        return if (readingText.isNotEmpty() && before.endsWith(readingText)) {
            before.dropLast(readingText.length).takeLast(maxLength)
        } else {
            before.takeLast(maxLength)
        }
    }

    // ------------------------------------------------------------------
    // 表示
    // ------------------------------------------------------------------

    private fun render(candidatesChanged: Boolean = true) {
        val ic = currentInputConnection
        when (state) {
            ComposingState.IDLE -> {
                ic?.finishComposingText()
                if (candidatesChanged) candidateView?.clear()
            }

            ComposingState.COMPOSING -> {
                ic?.setComposingText(livePreview ?: reading.toString(), 1)
                if (candidatesChanged) candidateView?.clear()
            }

            ComposingState.CONVERTING -> {
                val shown = candidates.getOrNull(selectedIndex) ?: reading.toString()
                ic?.setComposingText(shown, 1)
                if (candidatesChanged) {
                    candidateView?.submit(candidates, selectedIndex)
                } else {
                    candidateView?.updateSelection(selectedIndex)
                }
            }
        }
    }

    private fun resetComposing(finishComposing: Boolean) {
        convertGeneration++
        cancelLiveConversion()
        closeCorrection()
        livePreview = null
        reading.setLength(0)
        candidates = emptyList()
        selectedIndex = 0
        state = ComposingState.IDLE
        candidateView?.clear()
        if (finishComposing) {
            currentInputConnection?.finishComposingText()
        }
    }

    // ------------------------------------------------------------------
    // ヘルパ
    // ------------------------------------------------------------------

    /**
     * 直前の 1 文字を変換する。文字の種類で濁点か小文字かを決める。
     *
     * - か/さ/た/は行（つ を除く）→ 濁点（か→が）
     * - それ以外（あ/や/わ行・つ）→ 小文字（あ→ぁ、つ→っ）
     * - すでに濁点・小文字なら元に戻す（同じキーでトグル）
     */
    private fun transformTarget(ch: Char, handakuten: Boolean = false): Char? {
        if (handakuten) {
            // ダブルタップ: は →(1回目) ば →(2回目) ぱ になるよう、
            // 濁点を一度外してから半濁点を当てる。
            DAKUTEN_INVERSE[ch]?.let { base -> HANDAKUTEN_MAP[base]?.let { return it } }
            HANDAKUTEN_INVERSE[ch]?.let { return it }
            HANDAKUTEN_MAP[ch]?.let { return it }
            return null
        }
        SMALL_INVERSE[ch]?.let { return it }
        DAKUTEN_INVERSE[ch]?.let { return it }
        HANDAKUTEN_INVERSE[ch]?.let { return it }
        if (ch in SMALL_MAP) return SMALL_MAP[ch]
        if (ch != 'つ' && ch != 'ツ') DAKUTEN_MAP[ch]?.let { return it }
        return null
    }

    override fun onHandakutenPrevious() = transformPrevious(handakuten = true)

    override fun onTransformPrevious() = transformPrevious(handakuten = false)

    private fun transformPrevious(handakuten: Boolean) {
        when (state) {
            ComposingState.COMPOSING -> {
                if (reading.isEmpty()) return
                val last = reading[reading.length - 1]
                val next = transformTarget(last, handakuten) ?: return
                reading.setCharAt(reading.length - 1, next)
                candidates = emptyList()
                selectedIndex = 0
                livePreview = null
                render()
                scheduleLiveConversion()
            }

            ComposingState.CONVERTING -> {
                commitCurrent()
                transformCommitted(handakuten)
            }

            ComposingState.IDLE -> transformCommitted(handakuten)
        }
    }

    /** 確定済みテキストの直前 1 文字を置き換える（未確定の読みが無いとき）。 */
    private fun transformCommitted(handakuten: Boolean) {
        val ic = currentInputConnection ?: return
        val before = ic.getTextBeforeCursor(1, 0)?.toString().orEmpty()
        if (before.length != 1) return
        val next = transformTarget(before[0], handakuten) ?: return
        ic.deleteSurroundingText(1, 0)
        ic.commitText(next.toString(), 1)
    }

    private fun applyMark(map: Map<Char, Char>, fallback: Char) {
        if (reading.isEmpty()) {
            reading.append(fallback)
            return
        }
        val lastIndex = reading.length - 1
        val replaced = map[reading[lastIndex]]
        if (replaced != null) {
            reading.setCharAt(lastIndex, replaced)
        } else {
            reading.append(fallback)
        }
    }

    private fun insertText(text: String) {
        currentInputConnection?.commitText(text, 1)
    }

    private fun deleteOneCharBeforeCursor() {
        val ic: InputConnection = currentInputConnection ?: return
        if (ic.deleteSurroundingText(1, 0)) return
        ic.sendKeyEvent(KeyEvent(KeyEvent.ACTION_DOWN, KeyEvent.KEYCODE_DEL))
        ic.sendKeyEvent(KeyEvent(KeyEvent.ACTION_UP, KeyEvent.KEYCODE_DEL))
    }

    /** 空白は常に半角。変換キーと同居するため、未確定時は変換として働く。 */
    private fun spaceForMode(): String = " "

    private fun dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()

    private companion object {
        /** BERT に渡す文脈の最大文字数（設計書 §5.1: 直前 64 文字）。 */
        const val CONTEXT_MAX_LENGTH = 64

        /** 入力が止まってからライブ変換を発火するまでの時間。 */
        const val LIVE_CONVERT_DELAY_MS = 500L

        val DAKUTEN_MAP: Map<Char, Char> = buildMap {
            val pairs = "かがきぎくぐけげこごさざしじすずせぜそぞただちぢつづてでとどはばひびふぶへべほぼ"
            for (i in pairs.indices step 2) {
                put(pairs[i], pairs[i + 1])
            }
            put('う', 'ゔ')
        }

        val HANDAKUTEN_MAP: Map<Char, Char> = buildMap {
            val pairs = "はぱひぴふぷへぺほぽ"
            for (i in pairs.indices step 2) {
                put(pairs[i], pairs[i + 1])
            }
        }

        /** 小書きにできるかな。 */
        val SMALL_MAP: Map<Char, Char> = buildMap {
            val pairs = "あぁいぃうぅえぇおぉつっやゃゆゅよょわゎ" +
                "アァイィウゥエェオォツッヤャユュヨョワヮ"
            for (i in pairs.indices step 2) {
                put(pairs[i], pairs[i + 1])
            }
        }

        /** 濁点を外す。 */
        val DAKUTEN_INVERSE: Map<Char, Char> = buildMap {
            DAKUTEN_MAP.forEach { (base, voiced) -> put(voiced, base) }
        }

        /** 半濁点を外す。 */
        val HANDAKUTEN_INVERSE: Map<Char, Char> = buildMap {
            HANDAKUTEN_MAP.forEach { (base, semi) -> put(semi, base) }
        }

        /** 小文字を戻す。 */
        val SMALL_INVERSE: Map<Char, Char> = buildMap {
            SMALL_MAP.forEach { (base, small) -> put(small, base) }
        }
    }
}
