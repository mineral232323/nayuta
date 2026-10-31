package com.nayuta.ime

import android.content.Context
import android.content.SharedPreferences

/**
 * アプリの設定。SharedPreferences に保存する。
 *
 * IME サービスと設定画面の両方から読むので、[init] を一度呼んでから使う。
 * 未初期化のときは既定値を返す（落ちない）。
 */
object NayutaSettings {

    // ---- キー ----
    const val KEY_LIVE_DELAY = "live_delay_ms"
    const val KEY_ENGLISH = "english_convert"
    const val KEY_KEY_HEIGHT = "key_height"
    const val KEY_VIBRATE = "vibrate"
    const val KEY_THEME = "theme"
    const val KEY_FLICK_HINT = "flick_hint"

    // ---- 既定値 ----
    const val DEFAULT_LIVE_DELAY = 500
    const val DEFAULT_ENGLISH = true
    const val DEFAULT_KEY_HEIGHT = 1        // 0=小 / 1=中 / 2=大
    const val DEFAULT_VIBRATE = true
    const val DEFAULT_THEME = 0             // 0=システム / 1=ライト / 2=ダーク
    const val DEFAULT_FLICK_HINT = 0        // 0=標準 / 1=小さめ / 2=非表示

    /** ライブ変換の待ち時間の選択肢（0 はオフ）。 */
    val LIVE_DELAY_OPTIONS = intArrayOf(0, 300, 500, 800, 1000)

    /** キーの高さ（dp）。 */
    private val KEY_HEIGHT_DP = floatArrayOf(44f, 50f, 58f)

    private const val PREFS = "nayuta_settings"

    @Volatile
    private var prefs: SharedPreferences? = null

    fun init(context: Context) {
        if (prefs != null) return
        prefs = context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
    }

    private fun p(): SharedPreferences? = prefs

    fun register(context: Context, listener: SharedPreferences.OnSharedPreferenceChangeListener) {
        init(context)
        prefs?.registerOnSharedPreferenceChangeListener(listener)
    }

    fun unregister(listener: SharedPreferences.OnSharedPreferenceChangeListener) {
        prefs?.unregisterOnSharedPreferenceChangeListener(listener)
    }

    // ---- 読み書き ----

    /** 0 のときはライブ変換オフ。 */
    var liveDelayMs: Int
        get() = p()?.getInt(KEY_LIVE_DELAY, DEFAULT_LIVE_DELAY) ?: DEFAULT_LIVE_DELAY
        set(value) = p()?.edit()?.putInt(KEY_LIVE_DELAY, value)?.apply() ?: Unit

    /** カタカナ外来語を英単語にするか。 */
    var englishEnabled: Boolean
        get() = p()?.getBoolean(KEY_ENGLISH, DEFAULT_ENGLISH) ?: DEFAULT_ENGLISH
        set(value) = p()?.edit()?.putBoolean(KEY_ENGLISH, value)?.apply() ?: Unit

    /** キーの高さの段階（0=小 / 1=中 / 2=大）。 */
    var keyHeightIndex: Int
        get() = (p()?.getInt(KEY_KEY_HEIGHT, DEFAULT_KEY_HEIGHT) ?: DEFAULT_KEY_HEIGHT)
            .coerceIn(0, KEY_HEIGHT_DP.size - 1)
        set(value) = p()?.edit()?.putInt(KEY_KEY_HEIGHT, value.coerceIn(0, KEY_HEIGHT_DP.size - 1))
            ?.apply() ?: Unit

    /** 1 行ぶんのキーの高さ（dp）。 */
    val rowUnitDp: Float get() = KEY_HEIGHT_DP[keyHeightIndex]

    var vibrate: Boolean
        get() = p()?.getBoolean(KEY_VIBRATE, DEFAULT_VIBRATE) ?: DEFAULT_VIBRATE
        set(value) = p()?.edit()?.putBoolean(KEY_VIBRATE, value)?.apply() ?: Unit

    /** フリックのヒント表示: 0=標準 / 1=小さめ / 2=非表示。 */
    var flickHintMode: Int
        get() = (p()?.getInt(KEY_FLICK_HINT, DEFAULT_FLICK_HINT) ?: DEFAULT_FLICK_HINT)
            .coerceIn(0, 2)
        set(value) = p()?.edit()?.putInt(KEY_FLICK_HINT, value.coerceIn(0, 2))?.apply() ?: Unit

    /** 0=システム / 1=ライト / 2=ダーク。 */
    var themeMode: Int
        get() = p()?.getInt(KEY_THEME, DEFAULT_THEME) ?: DEFAULT_THEME
        set(value) = p()?.edit()?.putInt(KEY_THEME, value)?.apply() ?: Unit
}
