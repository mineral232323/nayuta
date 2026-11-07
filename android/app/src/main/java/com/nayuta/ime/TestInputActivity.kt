package com.nayuta.ime

import android.os.Bundle
import android.text.InputType
import android.util.Log
import android.view.Gravity
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.nayuta.ime.converter.Converters

/**
 * キーボードの動作確認用の入力画面。
 *
 * 実機テストでメッセージアプリなどを使うと、連絡先や電話番号といった
 * 個人情報が画面に写り込む。テストは必ずこの画面で行う。
 *
 * 起動: アプリのアイコン、または
 *   adb shell am start -n com.nayuta.ime/.TestInputActivity
 */
class TestInputActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val pad = (resources.displayMetrics.density * 16).toInt()

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, pad * 3, pad, pad)
            // 背景色はテーマに任せる（ダーク/ライト両対応のため固定色をやめた）
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
        root.setOnApplyWindowInsetsListener { v, insets ->
            val bars = insets.getInsets(android.view.WindowInsets.Type.systemBars())
            v.setPadding(pad, bars.top + pad, pad, pad)
            insets
        }
        setContentView(root)

        // 実機検証用の簡易ベンチマーク:
        //   adb shell am start -n com.nayuta.ime/.TestInputActivity         //       --es bench "きしゃする,にほんごのへんかんせいど" --ei bench_runs 3
        // 読みを変換して logcat (タグ NayutaBench) に結果と所要時間を出す。
        val bench = intent.getStringExtra("bench")
        if (!bench.isNullOrBlank()) {
            val readings = bench.split(",").map { it.trim() }.filter { it.isNotEmpty() }
            val runs = intent.getIntExtra("bench_runs", 1).coerceIn(1, 50)
            val appContext = applicationContext
            // 初回 convert は辞書・bigram・BERT のロードを含むので UI スレッドを塞がない
            Thread {
                val engine = Converters.get(appContext)
                val shown = StringBuilder()
                Log.d("NayutaBench", "start n=${readings.size} runs=$runs")
                for (reading in readings) {
                    var best: String? = null
                    for (i in 0 until runs) {
                        val t0 = System.nanoTime()
                        val res = engine.convert(reading, "")
                        val ms = (System.nanoTime() - t0) / 1_000_000.0
                        if (i == 0) best = res.firstOrNull()
                        Log.d(
                            "NayutaBench",
                            "reading=$reading run=$i best=${res.firstOrNull()} " +
                                "n=${res.size} ms=%.1f".format(ms),
                        )
                    }
                    shown.append(reading).append(" -> ").append(best).append("\n")
                }
                Log.d("NayutaBench", "done")
                runOnUiThread { field.setText(shown.toString()) }
            }.start()
        }

        field.requestFocus()
    }
}
