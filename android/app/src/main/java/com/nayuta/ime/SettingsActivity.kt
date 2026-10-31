package com.nayuta.ime

import android.content.Intent
import android.os.Bundle
import android.view.Gravity
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.RadioGroup
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.appbar.MaterialToolbar
import com.google.android.material.button.MaterialButton
import com.google.android.material.card.MaterialCardView
import com.google.android.material.color.MaterialColors
import com.google.android.material.divider.MaterialDivider
import com.google.android.material.materialswitch.MaterialSwitch
import com.google.android.material.radiobutton.MaterialRadioButton
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
                addView(label("ライブ変換（入力が止まってから変換するまで）"))
                addView(
                    radioOptions(
                        items = listOf("オフ", "300ms", "500ms", "800ms", "1秒"),
                        selected = NayutaSettings.LIVE_DELAY_OPTIONS
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
                addView(label("キーの高さ"))
                addView(
                    radioOptions(listOf("小", "中", "大"), NayutaSettings.keyHeightIndex) { i ->
                        NayutaSettings.keyHeightIndex = i
                    },
                )
                addView(divider())
                addView(label("フリックのヒント（キー上の小さい文字）"))
                addView(
                    radioOptions(listOf("標準", "小さめ", "非表示"), NayutaSettings.flickHintMode) { i ->
                        NayutaSettings.flickHintMode = i
                    },
                )
                addView(divider())
                addView(
                    switchRow("振動フィードバック", NayutaSettings.vibrate) { on ->
                        NayutaSettings.vibrate = on
                    },
                )
                addView(divider())
                addView(label("外観"))
                addView(
                    radioOptions(listOf("システム", "ライト", "ダーク"), NayutaSettings.themeMode) { i ->
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
     * ラジオの選択肢。行全体をタップできるようにし、MaterialRadioButton を使う。
     * RadioGroup は子の ID で排他制御するので、子には一意な ID を振る。
     */
    private fun radioOptions(
        items: List<String>,
        selected: Int,
        onChange: (Int) -> Unit,
    ): RadioGroup {
        val ids = items.indices.map { android.view.View.generateViewId() }
        val group = RadioGroup(this).apply {
            orientation = RadioGroup.VERTICAL
        }
        items.forEachIndexed { i, text ->
            val row = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                minimumHeight = dp(48)
                isClickable = true
                isFocusable = true
                setOnClickListener { group.check(ids[i]) }
                addView(
                    MaterialRadioButton(this@SettingsActivity).apply {
                        id = ids[i]
                        isChecked = i == selected
                        isClickable = false
                        isFocusable = false
                    },
                )
                addView(
                    TextView(this@SettingsActivity).apply {
                        this.text = text
                        setTextAppearance(
                            com.google.android.material.R.style.TextAppearance_Material3_BodyMedium,
                        )
                        setTextColor(
                            MaterialColors.getColor(
                                this,
                                com.google.android.material.R.attr.colorOnSurface,
                            ),
                        )
                        setPadding(dp(4), 0, 0, 0)
                    },
                    LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f),
                )
            }
            // 行はカード幅いっぱいに広げる（ラベル部分もタップできるように）
            group.addView(
                row,
                RadioGroup.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ),
            )
        }
        group.setOnCheckedChangeListener { _, checkedId ->
            val index = ids.indexOf(checkedId)
            if (index >= 0) onChange(index)
        }
        return group
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
