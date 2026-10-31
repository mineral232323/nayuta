package com.nayuta.ime

import android.app.Activity
import android.os.Bundle
import android.text.InputType
import android.view.Gravity
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView

/**
 * キーボードの動作確認用の入力画面。
 *
 * 実機テストでメッセージアプリなどを使うと、連絡先や電話番号といった
 * 個人情報が画面に写り込む。テストは必ずこの画面で行う。
 *
 * 起動: アプリのアイコン、または
 *   adb shell am start -n com.nayuta.ime/.TestInputActivity
 */
class TestInputActivity : Activity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val pad = (resources.displayMetrics.density * 16).toInt()

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, pad * 3, pad, pad)
            setBackgroundColor(0xFFF7F8FA.toInt())
        }

        root.addView(
            TextView(this).apply {
                text = "Nayuta テスト入力"
                textSize = 20f
            },
        )
        root.addView(
            TextView(this).apply {
                text = "この画面はアプリ内のテスト用です。個人情報は含みません。"
                textSize = 12f
                setPadding(0, pad / 2, 0, 0)
            },
        )

        val field = EditText(this).apply {
            hint = "ここに入力してください"
            gravity = Gravity.TOP or Gravity.START
            inputType = InputType.TYPE_CLASS_TEXT or
                InputType.TYPE_TEXT_FLAG_MULTI_LINE
            minLines = 6
            setPadding(pad, pad, pad, pad)
            setBackgroundColor(0xFFFFFFFF.toInt())
        }
        root.addView(
            field,
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                0,
                1f,
            ).apply { topMargin = pad },
        )

        // 画面下に余白を入れ、キーボードと重ならないようにする
        setContentView(root)
        field.requestFocus()
    }
}
