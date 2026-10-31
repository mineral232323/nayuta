package com.nayuta.ime

import android.content.Intent
import android.os.Bundle
import android.view.Gravity
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.appbar.MaterialToolbar
import com.google.android.material.button.MaterialButton
import com.google.android.material.card.MaterialCardView
import com.google.android.material.color.MaterialColors
import com.google.android.material.divider.MaterialDivider
import com.google.android.material.materialswitch.MaterialSwitch
import com.google.android.material.textfield.MaterialAutoCompleteTextView
import com.google.android.material.textfield.TextInputLayout
import com.nayuta.ime.converter.dict.UserDictionary

/**
 * 設定画面（Material 3）。
 *
 * - MaterialToolbar に「Nayuta 設定」
 * - セクション見出し + アウトラインカード
 * - カード内は 48〜56dp の行（ラベル + コントロール）、行全体がタップ可能
 */
class SettingsActivity : AppCompatActivity() {

    private lateinit var userDictCount: TextView
    private val density get() = resources.displayMetrics.density

    private fun dp(v: Int): Int = (v * density).toInt()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        NayutaSettings.init(this)

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
        }

        root.addView(
            MaterialToolbar(this).apply {
                title = "Nayuta 設定"
            },
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                dp(64),
            ),
        )

        val content = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(8), dp(16), dp(24))
        }

        // ---------------- 変換 ----------------
        content.addView(sectionHeader("変換"))
        content.addView(
            card {
                addView(
                    dropdown(
                        "ライブ変換（入力が止まってから変換するまで）",
                        listOf("オフ", "300ms", "500ms", "800ms", "1秒"),
                        NayutaSettings.LIVE_DELAY_OPTIONS
                            .indexOf(NayutaSettings.liveDelayMs).coerceAtLeast(0),
                    ) { i -> NayutaSettings.liveDelayMs = NayutaSettings.LIVE_DELAY_OPTIONS[i] },
                )
                addView(divider())
                addView(
                    switchRow("カタカナ語を英語に変換", NayutaSettings.englishEnabled) { on ->
                        NayutaSettings.englishEnabled = on
                    },
                )
            },
        )

        // ---------------- キーボード ----------------
        content.addView(sectionHeader("キーボード"))
        content.addView(
            card {
                addView(
                    dropdown("キーの高さ", listOf("小", "中", "大"), NayutaSettings.keyHeightIndex) { i ->
                        NayutaSettings.keyHeightIndex = i
                    },
                )
                addView(divider())
                addView(
                    dropdown(
                        "フリックのヒント（キー上の小さい文字）",
                        listOf("標準", "小さめ", "非表示"),
                        NayutaSettings.flickHintMode,
                    ) { i -> NayutaSettings.flickHintMode = i },
                )
                addView(divider())
                addView(
                    switchRow("振動フィードバック", NayutaSettings.vibrate) { on ->
                        NayutaSettings.vibrate = on
                    },
                )
                addView(divider())
                addView(
                    dropdown("外観", listOf("システム", "ライト", "ダーク"), NayutaSettings.themeMode) { i ->
                        NayutaSettings.themeMode = i
                    },
                )
            },
        )

        // ---------------- 辞書 ----------------
        content.addView(sectionHeader("辞書"))
        content.addView(
            card {
                userDictCount = label("登録数: ${UserDictionary.get(applicationContext).size} 語")
                addView(userDictCount)
                addView(
                    MaterialButton(
                        this@SettingsActivity,
                        null,
                        com.google.android.material.R.attr.materialButtonOutlinedStyle,
                    ).apply {
                        text = "ユーザー辞書をすべて削除"
                        setOnClickListener {
                            UserDictionary.get(applicationContext).clear()
                            refreshUserDict()
                        }
                        layoutParams = LinearLayout.LayoutParams(
                            ViewGroup.LayoutParams.MATCH_PARENT,
                            ViewGroup.LayoutParams.WRAP_CONTENT,
                        ).apply { topMargin = dp(8); bottomMargin = dp(8) }
                    },
                )
            },
        )

        // ---------------- その他 ----------------
        content.addView(sectionHeader("その他"))
        content.addView(
            card {
                addView(
                    MaterialButton(
                        this@SettingsActivity,
                        null,
                        com.google.android.material.R.attr.materialButtonOutlinedStyle,
                    ).apply {
                        text = "テスト入力画面を開く"
                        setOnClickListener {
                            startActivity(Intent(this@SettingsActivity, TestInputActivity::class.java))
                        }
                        layoutParams = LinearLayout.LayoutParams(
                            ViewGroup.LayoutParams.MATCH_PARENT,
                            ViewGroup.LayoutParams.WRAP_CONTENT,
                        ).apply { topMargin = dp(8); bottomMargin = dp(8) }
                    },
                )
                addView(divider())
                addView(caption("バージョン ${versionName()}"))
                addView(caption("入力内容は端末外に送信しません。"))
            },
        )

        root.addView(ScrollView(this).apply { addView(content) })
        setContentView(root)
    }

    override fun onResume() {
        super.onResume()
        refreshUserDict()
    }

    private fun refreshUserDict() {
        userDictCount.text = "登録数: ${UserDictionary.get(applicationContext).size} 語"
    }

    // ------------------------------------------------------------------
    // Material 部品
    // ------------------------------------------------------------------

    private fun sectionHeader(text: String): TextView = TextView(this).apply {
        this.text = text
        setTextAppearance(com.google.android.material.R.style.TextAppearance_Material3_TitleSmall)
        setTextColor(MaterialColors.getColor(this, com.google.android.material.R.attr.colorPrimary))
        setPadding(dp(4), dp(20), 0, dp(8))
    }

    private fun card(block: LinearLayout.() -> Unit): MaterialCardView {
        val column = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(8), dp(16), dp(8))
            block()
        }
        return MaterialCardView(this).apply {
            radius = dp(16).toFloat()
            strokeWidth = dp(1)
            setStrokeColor(
                MaterialColors.getColor(this, com.google.android.material.R.attr.colorOutlineVariant),
            )
            setCardBackgroundColor(
                MaterialColors.getColor(this, com.google.android.material.R.attr.colorSurface),
            )
            elevation = 0f
            addView(column)
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ).apply { bottomMargin = dp(8) }
        }
    }

    private fun label(text: String): TextView = TextView(this).apply {
        this.text = text
        setTextAppearance(com.google.android.material.R.style.TextAppearance_Material3_BodyLarge)
        setTextColor(
            MaterialColors.getColor(this, com.google.android.material.R.attr.colorOnSurface),
        )
        setPadding(0, dp(16), 0, dp(4))
    }

    private fun caption(text: String): TextView = TextView(this).apply {
        this.text = text
        setTextAppearance(com.google.android.material.R.style.TextAppearance_Material3_BodySmall)
        setTextColor(
            MaterialColors.getColor(this, com.google.android.material.R.attr.colorOnSurfaceVariant),
        )
        setPadding(0, dp(10), 0, dp(10))
    }

    private fun divider(): MaterialDivider = MaterialDivider(this).apply {
        setDividerColor(
            MaterialColors.getColor(this, com.google.android.material.R.attr.colorOutlineVariant),
        )
        setDividerInsetStart(dp(0))
        setDividerInsetEnd(dp(0))
    }

    /**
     * プルダウン（Material の Exposed Dropdown Menu）。
     * ラベルを上に出し、その下に選択式のフィールドを置く。
     */
    private fun dropdown(
        title: String,
        items: List<String>,
        selected: Int,
        onChange: (Int) -> Unit,
    ): LinearLayout = LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        addView(label(title))

        val field = MaterialAutoCompleteTextView(this@SettingsActivity).apply {
            inputType = android.text.InputType.TYPE_NULL
            isCursorVisible = false
            setSimpleItems(items.toTypedArray())
            setText(items[selected.coerceIn(0, items.size - 1)], false)
            setOnItemClickListener { _, _, position, _ -> onChange(position) }
            // TYPE_NULL だとタップで自動的に開かないので、フォーカスを外して
            // クリックで開くようにする（欄のどこを押しても開く）
            isFocusable = false
            isClickable = true
            setOnClickListener { showDropDown() }
        }
        val wrap = TextInputLayout(
            this@SettingsActivity,
            null,
            com.google.android.material.R.attr.textInputOutlinedExposedDropdownMenuStyle,
        ).apply {
            setEndIconMode(TextInputLayout.END_ICON_DROPDOWN_MENU)
            addView(field)
            setOnClickListener { field.showDropDown() }
        }
        addView(
            wrap,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ).apply { bottomMargin = dp(12) },
        )
    }

    private fun switchRow(text: String, checked: Boolean, onChange: (Boolean) -> Unit): LinearLayout =
        LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            minimumHeight = dp(56)
            addView(
                TextView(this@SettingsActivity).apply {
                    this.text = text
                    setTextAppearance(
                        com.google.android.material.R.style.TextAppearance_Material3_BodyLarge,
                    )
                    setTextColor(
                        MaterialColors.getColor(
                            this,
                            com.google.android.material.R.attr.colorOnSurface,
                        ),
                    )
                },
                LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f),
            )
            addView(
                MaterialSwitch(this@SettingsActivity).apply {
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
