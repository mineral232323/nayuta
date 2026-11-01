package com.nayuta.ime

import android.inputmethodservice.InputMethodService
import android.os.Handler
import android.text.SpannableStringBuilder
import android.text.Spanned
import android.text.style.BackgroundColorSpan
import android.text.style.UnderlineSpan
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

    /** 英字の大文字入力モード（1 文字入力すると戻る）。 */
    private var shiftOn = false

    // ---- 単語ごとの変換モード ----
    private var segMode = false
    private var segReading = ""
    private val segBounds = ArrayList<Int>()
    private val segCands = ArrayList<List<String>>()
    private val segSel = ArrayList<Int>()
    private val segTouched = ArrayList<Boolean>()
    private var segActive = 0
    private var segGeneration = 0

    /** 直前に確定した語（訂正の対象）。 */
    private var selectedIndex = 0
    private var state: ComposingState = ComposingState.IDLE
    private var mode: KeyboardMode = KeyboardMode.KANA

    /** 変換要求の世代番号。古い非同期結果を捨てるために使う。 */
    private var convertGeneration = 0

    // ------------------------------------------------------------------
    // ライフサイクル
    // ------------------------------------------------------------------

    /** 設定が変わったらキーボードを作り直す（高さ・振動・外観を反映）。 */
    private val prefListener =
        android.content.SharedPreferences.OnSharedPreferenceChangeListener { _, _ ->
            uiHandler.post {
                KeyboardColors.resolve(this)
                setInputView(onCreateInputView())
            }
        }

    override fun onCreate() {
        super.onCreate()
        NayutaSettings.init(this)
        com.nayuta.ime.NayutaSettings.register(this, prefListener)
        converter = Converters.get(this)
        convertExecutor = Executors.newSingleThreadExecutor { runnable ->
            Thread(runnable, "nayuta-convert")
        }
    }

    override fun onDestroy() {
        convertExecutor?.shutdownNow()
        convertExecutor = null
        uiHandler.removeCallbacksAndMessages(null)
        com.nayuta.ime.NayutaSettings.unregister(prefListener)
        super.onDestroy()
    }

    override fun onCreateInputView(): View {
        android.util.Log.d("NayutaInset", "onCreateInputView called")
        // 端末のダークモード設定を反映してから組む
        KeyboardColors.resolve(this)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(KeyboardColors.keyboardBackground)
        }

        // 端末によっては IME のウィンドウがナビゲーションバー（ジェスチャーバー）
        // の下まで伸びてしまい、最下段のキーが隠れる。システムバーの
        // インセットぶんだけ内側に寄せる（インセットが 0 の端末では何も起きない）。
        // 端末によっては IME のウィンドウがナビゲーションバー（ジェスチャーバー）
        // の裏まで伸びて最下段のキーが隠れる。一方、窓がバーの上で止まっている
        // 端末（AQUOS など）では同じインセット値が返るため、そのまま足すと
        // 余白が二重になる。実際の画面座標で「重なっている分だけ」を補正する。
        root.addOnAttachStateChangeListener(object : View.OnAttachStateChangeListener {
            override fun onViewAttachedToWindow(v: View) {
                v.post {
                    val ins = v.rootWindowInsets ?: return@post
                    val navBottom = if (android.os.Build.VERSION.SDK_INT >= 30) {
                        ins.getInsets(android.view.WindowInsets.Type.systemBars()).bottom
                    } else {
                        @Suppress("DEPRECATION")
                        ins.systemWindowInsetBottom
                    }
                    val loc = IntArray(2)
                    v.getLocationOnScreen(loc)
                    val viewBottom = loc[1] + v.height
                    val screenH = resources.displayMetrics.heightPixels
                    val navTop = screenH - navBottom
                    val overlap = viewBottom - navTop
                    android.util.Log.d(
                        "NayutaInset",
                        "navBottom=$navBottom viewBottom=$viewBottom navTop=$navTop overlap=$overlap",
                    )
                    if (overlap > 0) v.setPadding(0, 0, 0, overlap)
                }
            }

            override fun onViewDetachedFromWindow(v: View) {}
        })

        val candidates = CandidateView(this).apply {
            onCandidateSelected = { index, _ -> selectCandidate(index) }
        }
        val keyboard = FlickKeyboardView(this).apply {
            listener = this@NayutaImeService
            setLayout(layoutFor(mode))
        }

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
        return root
    }

    override fun onConfigurationChanged(newConfig: android.content.res.Configuration) {
        super.onConfigurationChanged(newConfig)
        val wasDark = KeyboardColors.isDark
        if (KeyboardColors.resolve(this) == wasDark) return
        // テーマが変わったときだけ塗り直す
        candidateView?.applyTheme()
        keyboardView?.applyTheme()
        (keyboardView?.parent as? View)?.setBackgroundColor(KeyboardColors.keyboardBackground)
    }

    override fun onStartInputView(info: EditorInfo?, restarting: Boolean) {
        super.onStartInputView(info, restarting)
        resetComposing(finishComposing = true)
        keyboardView?.setLayout(layoutFor(mode))
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
        if (segMode) exitSegmentMode()
        if (state == ComposingState.CONVERTING) commitCurrent()

        when (text) {
            KanaMarks.DAKUTEN -> applyMark(DAKUTEN_MAP, KanaMarks.DAKUTEN_CHAR)
            KanaMarks.HANDAKUTEN -> applyMark(HANDAKUTEN_MAP, KanaMarks.HANDAKUTEN_CHAR)
            else -> reading.append(text)
        }

        if (mode == KeyboardMode.ALPHABET && shiftOn) {
            shiftOn = false
            keyboardView?.setLayout(layoutFor(mode))
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
        if (segMode) return
        if (reading.isEmpty()) return
        val target = reading.toString()
        val delay = NayutaSettings.liveDelayMs
        if (delay <= 0) return
        val job = Runnable { startLiveConversion(target) }
        liveJob = job
        uiHandler.postDelayed(job, delay.toLong())
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
        if (segMode) exitSegmentMode()
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
        if (segMode) {
            cycleSegmentCandidate()
            return
        }
        when (state) {
            ComposingState.CONVERTING -> {
                if (candidates.size > 1) {
                    selectedIndex = (selectedIndex + 1) % candidates.size
                    render(candidatesChanged = false)
                }
            }

            ComposingState.COMPOSING -> startSegmentMode()
            ComposingState.IDLE -> Unit
        }
    }

    override fun onSpace() {
        if (segMode) {
            cycleSegmentCandidate()
            return
        }
        when (state) {
            ComposingState.CONVERTING -> {
                commitCurrent()
                insertText(spaceForMode())
            }

            // かな入力中は空白キー = 変換（単語ごとの変換モードに入る）
            ComposingState.COMPOSING -> startSegmentMode()
            ComposingState.IDLE -> insertText(spaceForMode())
        }
    }

    override fun onCursorMove(left: Boolean) {
        if (segMode) {
            adjustSegmentBoundary(left)
            return
        }
        val ic = currentInputConnection ?: return
        // 変換中なら確定してから動かす（編集中の読みを壊さない）
        if (state != ComposingState.IDLE) commitCurrent()
        val code = if (left) KeyEvent.KEYCODE_DPAD_LEFT else KeyEvent.KEYCODE_DPAD_RIGHT
        ic.sendKeyEvent(KeyEvent(KeyEvent.ACTION_DOWN, code))
        ic.sendKeyEvent(KeyEvent(KeyEvent.ACTION_UP, code))
    }

    /**
     * ✓ / Enter。
     *
     * - 未確定（入力中・変換中・ライブ変換のプレビュー中）なら **確定だけ** 行う。
     *   ここで改行まで入れてしまうと「変換したら改行された」ことになる。
     * - 何も未確定でないときだけ Enter。単一行の入力欄（検索・送信など）は
     *   エディタアクションを実行し、複数行なら改行する。
     */
    override fun onEnter() {
        if (segMode) {
            confirmSegment()
            return
        }
        val ic = currentInputConnection ?: return

        if (state != ComposingState.IDLE) {
            commitCurrent()
            return
        }

        val imeOptions = currentInputEditorInfo?.imeOptions ?: 0
        // 複数行の入力欄は IME_FLAG_NO_ENTER_ACTION が立っていて、
        // Enter は「送信」ではなく「改行」として扱う必要がある。
        val newlineField = (imeOptions and EditorInfo.IME_FLAG_NO_ENTER_ACTION) != 0
        val action = imeOptions and EditorInfo.IME_MASK_ACTION
        if (!newlineField &&
            action != EditorInfo.IME_ACTION_NONE &&
            action != EditorInfo.IME_ACTION_UNSPECIFIED
        ) {
            ic.performEditorAction(action)
        } else {
            ic.sendKeyEvent(KeyEvent(KeyEvent.ACTION_DOWN, KeyEvent.KEYCODE_ENTER))
            ic.sendKeyEvent(KeyEvent(KeyEvent.ACTION_UP, KeyEvent.KEYCODE_ENTER))
        }
    }

    // ------------------------------------------------------------------
    // 単語ごとの変換モード
    // ------------------------------------------------------------------

    /** 保留中の読みを単語に区切り、先頭の語を選択状態にする。 */
    private fun startSegmentMode() {
        val target = reading.toString()
        android.util.Log.d("NayutaSeg", "start target=$target state=$state")
        if (target.isEmpty()) return
        val engine = converter ?: return
        val executor = convertExecutor ?: return
        cancelLiveConversion()
        val gen = ++segGeneration
        val ctx = contextBeforeCursor()

        executor.execute {
            val segs = try {
                engine.segments(target, ctx)
            } catch (e: Exception) {
                emptyList()
            }
            uiHandler.post {
                if (gen != segGeneration) return@post
                if (reading.toString() != target) return@post

                segBounds.clear()
                segBounds.add(0)
                for (sg in segs) {
                    if (sg.start == segBounds.last() && sg.end > sg.start) segBounds.add(sg.end)
                }
                if (segBounds.last() != target.length) segBounds.add(target.length)
                android.util.Log.d("NayutaSeg", "segs=${segs.size} bounds=$segBounds")
                if (segBounds.size < 2) return@post

                segReading = target
                segCands.clear(); segSel.clear(); segTouched.clear()
                for (i in 0 until segBounds.size - 1) {
                    segCands.add(candidatesOf(target, i))
                    segSel.add(0)
                    segTouched.add(false)
                }
                segActive = 0
                segMode = true
                livePreview = null
                renderSegmentMode()
            }
        }
    }

    private fun candidatesOf(target: String, i: Int): List<String> {
        val part = target.substring(segBounds[i], segBounds[i + 1])
        val c = converter?.candidatesFor(part).orEmpty()
        return if (c.isEmpty()) listOf(part) else c
    }

    /** 選択中の語だけ下線＋薄い背景でハイライトしてアプリ側に出す。 */
    private fun renderSegmentMode() {
        val ic = currentInputConnection ?: return
        if (!segMode || segCands.isEmpty()) return
        val sb = SpannableStringBuilder()
        for (i in segCands.indices) {
            val surface = segCands[i].getOrNull(segSel[i])
                ?: segReading.substring(segBounds[i], segBounds[i + 1])
            val start = sb.length
            sb.append(surface)
            if (i == segActive && sb.length > start) {
                sb.setSpan(
                    BackgroundColorSpan(0x330B57D0),
                    start, sb.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE,
                )
                sb.setSpan(UnderlineSpan(), start, sb.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
            }
        }
        ic.setComposingText(sb, 1)
        val cur = segCands[segActive]
        candidateView?.submit(cur, segSel[segActive].coerceIn(0, cur.size - 1))
    }

    /** 選択中の語の長さ（区切り）を 1 文字単位で変える。 */
    private fun adjustSegmentBoundary(left: Boolean) {
        android.util.Log.d("NayutaSeg", "adjust left=$left segMode=$segMode active=$segActive")
        if (!segMode || segCands.isEmpty()) return
        val i = segActive
        if (left) {
            if (segBounds[i + 1] - segBounds[i] > 1) {
                segBounds[i + 1] -= 1
                // 最後の語を縮めたときは、はみ出した分を新しい語として足す
                if (i == segCands.size - 1 && segBounds[i + 1] < segReading.length) {
                    segBounds.add(segReading.length)
                    segCands.add(emptyList())
                    segSel.add(0)
                    segTouched.add(false)
                }
                refreshSegment(i)
                if (i + 1 < segCands.size) refreshSegment(i + 1)
                segTouched[i] = true
                renderSegmentMode()
            } else if (i > 0) {
                segActive = i - 1
                renderSegmentMode()
            }
        } else {
            if (segBounds[i + 1] < segReading.length) {
                segBounds[i + 1] += 1
                refreshSegment(i)
                if (i + 1 < segCands.size) refreshSegment(i + 1)
                segTouched[i] = true
                // 次の語が空になったら詰める
                if (i + 1 < segCands.size && segBounds[i + 2] == segBounds[i + 1]) {
                    segBounds.removeAt(i + 1)
                    segCands.removeAt(i + 1)
                    segSel.removeAt(i + 1)
                    segTouched.removeAt(i + 1)
                }
                renderSegmentMode()
            } else if (i < segCands.size - 1) {
                segActive = i + 1
                renderSegmentMode()
            }
        }
    }

    private fun cycleSegmentCandidate() {
        if (segCands.isEmpty()) return
        val cur = segCands[segActive]
        if (cur.size > 1) {
            segSel[segActive] = (segSel[segActive] + 1) % cur.size
            segTouched[segActive] = true
        }
        renderSegmentMode()
    }

    private fun refreshSegment(i: Int) {
        if (i < 0 || i >= segCands.size) return
        segCands[i] = candidatesOf(segReading, i)
        segSel[i] = 0
    }

    /** 確定して次の語へ。最後の語なら全体を確定する。 */
    private fun confirmSegment() {
        if (!segMode || segCands.isEmpty()) return
        if (segActive < segCands.size - 1) {
            segActive++
            renderSegmentMode()
        } else {
            commitSegments()
        }
    }

    private fun commitSegments() {
        val ic = currentInputConnection
        // 確定すると文脈が変わるので、確定前の文脈を控えておく
        val ctx = contextBeforeCursor()
        val snapshot = segCands.indices.map { i ->
            Triple(
                segReading.substring(segBounds[i], segBounds[i + 1]),
                segCands[i],
                segSel[i],
            )
        }
        val out = buildString {
            for (i in segCands.indices) append(segCands[i].getOrNull(segSel[i]) ?: "")
        }
        if (ic != null && out.isNotEmpty()) ic.commitText(out, 1)

        // 1) 手で直した語だけユーザー辞書に覚える（丸暗記）
        val ud = UserDictionary.get(applicationContext)
        for (i in segCands.indices) {
            if (!segTouched.getOrElse(i) { false }) continue
            val r = segReading.substring(segBounds[i], segBounds[i + 1])
            val s = segCands[i].getOrNull(segSel[i]) ?: continue
            if (r.isNotEmpty() && s.isNotEmpty()) ud.add(r, s)
        }

        // 2) 薄い層のオンライン更新。確定した語すべてについて、
        //    「選んだ表記の確率を上げる」方向に w を動かす（メインスレッドを塞がない）
        val engine = converter
        val executor = convertExecutor
        if (engine != null && executor != null) {
            executor.execute {
                for ((r, cands, chosen) in snapshot) {
                    try {
                        engine.learn(ctx, r, cands, chosen)
                    } catch (e: Exception) {
                        android.util.Log.e("NayutaOnline", "learn failed", e)
                    }
                }
            }
        }

        exitSegmentMode()
        resetComposing(finishComposing = true)
    }

    private fun exitSegmentMode() {
        segMode = false
        segCands.clear(); segBounds.clear(); segSel.clear(); segTouched.clear()
        segActive = 0
    }

    /** 現在のモードに応じたレイアウト。英数は Shift 状態を反映する。 */
    private fun layoutFor(mode: KeyboardMode): com.nayuta.ime.keyboard.KeyboardLayout =
        if (mode == KeyboardMode.ALPHABET) {
            KeyboardLayouts.alphabet(shiftOn, NayutaSettings.numberRowEnabled)
        } else {
            KeyboardLayouts.forMode(mode)
        }

    override fun onShift() {
        shiftOn = !shiftOn
        if (mode == KeyboardMode.ALPHABET) keyboardView?.setLayout(layoutFor(mode))
    }

    private fun toast(message: String) {
        android.widget.Toast.makeText(this, message, android.widget.Toast.LENGTH_SHORT).show()
    }

    override fun onNumberToggle() {
        if (state != ComposingState.IDLE) commitCurrent()
        mode = if (mode == KeyboardMode.NUMBER) KeyboardMode.KANA else KeyboardMode.NUMBER
        keyboardView?.setLayout(layoutFor(mode))
    }

    /** 英字 → 数字・記号ページ。 */
    override fun onSymbolToggle() {
        if (state != ComposingState.IDLE) commitCurrent()
        mode = KeyboardMode.SYMBOL
        keyboardView?.setLayout(layoutFor(mode))
    }

    /** 数字・記号ページ → 英字。 */
    override fun onAlphabetToggle() {
        if (state != ComposingState.IDLE) commitCurrent()
        mode = KeyboardMode.ALPHABET
        keyboardView?.setLayout(layoutFor(mode))
    }

    override fun onKanaToggle() {
        if (state != ComposingState.IDLE) commitCurrent()
        mode = KeyboardMode.KANA
        keyboardView?.setLayout(layoutFor(mode))
    }

    override fun onModeToggle() {
        if (state != ComposingState.IDLE) commitCurrent()
        mode = mode.toggled()
        keyboardView?.setLayout(layoutFor(mode))
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
        // 単語モード: 候補をタップしたら、その語を選んで次の語へ進む
        if (segMode) {
            if (segCands.isEmpty()) return
            val cur = segCands[segActive]
            if (cur.isEmpty()) return
            segSel[segActive] = index.coerceIn(0, cur.size - 1)
            segTouched[segActive] = true
            confirmSegment()
            return
        }
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
        if (segMode) {
            renderSegmentMode()
            return
        }
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
        segGeneration++
        exitSegmentMode()
        cancelLiveConversion()
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
