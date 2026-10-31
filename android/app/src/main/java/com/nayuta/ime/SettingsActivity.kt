package com.nayuta.ime

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.view.Gravity
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.ScrollView
import android.widget.Switch
import android.widget.TextView
import com.nayuta.ime.converter.dict.UserDictionary

/**
 * 設定画面。
 *
 * AndroidX を使わずフレームワークのウィジェットだけで組んでいる
 * （このプロジェクトは依存を最小限に保つ方針のため）。
 * 変更は即座に SharedPreferences に書かれ、開いているキーボードにも反映される。
 */
class SettingsActivity : Activity() {

    private lateinit var userDictCount: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        NayutaSettings.init(this)

        val pad = (resources.displayMetrics.density * 20).toInt()
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, pad, pad, pad)
        }

        root.addView(
            TextView(this).apply {
                text = "Nayuta 設定"
                textSize = 22f
            },
        )

        // ---------------- 変換 ----------------
        root.addView(section("変換"))

        root.addView(label("ライブ変換（入力が止まってから変換するまで）"))
        root.addView(
            radioGroup(
                items = listOf("オフ", "300ms", "500ms", "800ms", "1秒"),
                selected = NayutaSettings.LIVE_DELAY_OPTIONS
                    .indexOf(NayutaSettings.liveDelayMs).coerceAtLeast(0),
            ) { i -> NayutaSettings.liveDelayMs = NayutaSettings.LIVE_DELAY_OPTIONS[i] },
        )

        root.addView(
            switchRow("カタカナ語を英語に変換", NayutaSettings.englishEnabled) { on ->
                NayutaSettings.englishEnabled = on
            },
        )

        // ---------------- キーボード ----------------
        root.addView(section("キーボード"))

        root.addView(label("キーの高さ"))
        root.addView(
            radioGroup(listOf("小", "中", "大"), NayutaSettings.keyHeightIndex) { i ->
                NayutaSettings.keyHeightIndex = i
            },
        )

        root.addView(
            switchRow("振動フィードバック", NayutaSettings.vibrate) { on ->
                NayutaSettings.vibrate = on
            },
        )

        root.addView(label("外観"))
        root.addView(
            radioGroup(listOf("システム", "ライト", "ダーク"), NayutaSettings.themeMode) { i ->
                NayutaSettings.themeMode = i
            },
        )

        // ---------------- 辞書 ----------------
        root.addView(section("辞書"))
        userDictCount = label("")
        root.addView(userDictCount)
        root.addView(
            Button(this).apply {
                text = "ユーザー辞書をすべて削除"
                setOnClickListener {
                    UserDictionary.get(applicationContext).clear()
                    refreshUserDict()
                }
            },
        )
        refreshUserDict()

        // ---------------- その他 ----------------
        root.addView(section("その他"))
        root.addView(
            Button(this).apply {
                text = "テスト入力画面を開く"
                setOnClickListener {
                    startActivity(Intent(this@SettingsActivity, TestInputActivity::class.java))
                }
            },
        )
        root.addView(label("バージョン ${versionName()}"))
        root.addView(label("ソース: この端末内で完結します。入力内容は送信しません。"))

        setContentView(ScrollView(this).apply { addView(root) })
    }

    override fun onResume() {
        super.onResume()
        refreshUserDict()
    }

    private fun refreshUserDict() {
        userDictCount.text = "登録数: ${UserDictionary.get(applicationContext).size} 語"
    }

    // ---------------- UI 部品 ----------------

    private fun section(title: String): TextView = TextView(this).apply {
        text = title
        textSize = 12f
        setPadding(0, (resources.displayMetrics.density * 24).toInt(), 0, 0)
        alpha = 0.7f
    }

    private fun label(text: String): TextView = TextView(this).apply {
        this.text = text
        textSize = 14f
        setPadding(0, (resources.displayMetrics.density * 12).toInt(), 0, 0)
    }

    private fun radioGroup(
        items: List<String>,
        selected: Int,
        onChange: (Int) -> Unit,
    ): RadioGroup = RadioGroup(this).apply {
        orientation = RadioGroup.VERTICAL
        items.forEachIndexed { i, text ->
            addView(
                RadioButton(this@SettingsActivity).apply {
                    this.text = text
                    isChecked = i == selected
                    setOnCheckedChangeListener { _, checked ->
                        if (checked) onChange(i)
                    }
                },
            )
        }
    }

    private fun switchRow(text: String, checked: Boolean, onChange: (Boolean) -> Unit): LinearLayout =
        LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, (resources.displayMetrics.density * 8).toInt(), 0, 0)
            addView(
                TextView(this@SettingsActivity).apply {
                    this.text = text
                    textSize = 14f
                },
                LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f),
            )
            addView(
                Switch(this@SettingsActivity).apply {
                    isChecked = checked
                    setOnCheckedChangeListener { _, on -> onChange(on) }
                },
            )
        }

    private fun versionName(): String = try {
        packageManager.getPackageInfo(packageName, 0).versionName ?: "-"
    } catch (e: Exception) {
        "-"
    }
}
