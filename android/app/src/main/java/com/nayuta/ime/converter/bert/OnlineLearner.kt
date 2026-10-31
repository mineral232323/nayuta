package com.nayuta.ime.converter.bert

import android.content.Context
import android.content.SharedPreferences

/**
 * 端末内オンライン学習。更新するのは**最後の薄い線形層だけ**。
 *
 * スコア = BERT の元スコア + Σ_d w_d · h_d · c_d
 *     h = BERT が出力した文脈ベクトル（スパンの平均）
 *     c = 候補表記の文字埋め込み平均
 *
 * 確定するたびに、選ばれた候補の確率を上げ・競合の確率を下げる方向へ
 * w（数百次元）だけを SGD で更新する。モデル本体は凍結したまま。
 */
class OnlineLearner private constructor(
    private val prefs: SharedPreferences,
    val hidden: Int,
) {

    /** 学習する重み。初期値は 0（＝学習前は元スコアと同じ振る舞い）。 */
    val w: FloatArray = FloatArray(hidden)

    /** これまでに何回更新したか（設定画面などで表示する用）。 */
    var updateCount: Int = 0
        private set

    init {
        load()
    }

    private fun load() {
        val raw = prefs.getString(KEY_W, "").orEmpty()
        if (raw.isNotEmpty()) {
            val parts = raw.split(',')
            for (i in 0 until minOf(parts.size, hidden)) {
                w[i] = parts[i].toFloatOrNull() ?: 0f
            }
        }
        updateCount = prefs.getInt(KEY_COUNT, 0)
    }

    private fun persist() {
        val sb = StringBuilder(w.size * 8)
        for (i in w.indices) {
            if (i > 0) sb.append(',')
            sb.append(w[i])
        }
        prefs.edit().putString(KEY_W, sb.toString()).putInt(KEY_COUNT, updateCount).apply()
    }

    /**
     * 1 回分の更新。
     *
     * @param phi   (K, H) の特徴（スパン方向の余弦 × 候補方向の余弦）
     * @param probs (K) 現在のモデルが出した確率
     * @param chosen 実際に確定した候補の添字
     */
    fun update(phi: Array<FloatArray>, probs: FloatArray, chosen: Int) {
        if (phi.isEmpty() || chosen !in phi.indices) return
        val grad = FloatArray(hidden)
        for (k in phi.indices) {
            val coef = if (k == chosen) probs[k] - 1f else probs[k]
            if (coef == 0f) continue
            val pk = phi[k]
            val n = minOf(hidden, pk.size)
            for (d in 0 until n) grad[d] += coef * pk[d]
        }
        // 勾配に L2 を足して降下。暴れないよう重みをクリップする。
        for (d in 0 until hidden) {
            w[d] -= LR * (grad[d] + WEIGHT_DECAY * w[d])
            if (w[d] > MAX_ABS) w[d] = MAX_ABS
            if (w[d] < -MAX_ABS) w[d] = -MAX_ABS
        }
        updateCount++
        persist()
    }

    fun save() = persist()

    fun reset() {
        for (i in w.indices) w[i] = 0f
        updateCount = 0
        persist()
    }

    companion object {
        private const val PREFS = "nayuta_online"
        private const val KEY_W = "w"
        private const val KEY_COUNT = "count"

        /** 学習率。 */
        private const val LR = 3.0f

        /** L2 正則化。 */
        private const val WEIGHT_DECAY = 1e-4f

        /** 重みの絶対値の上限。 */
        private const val MAX_ABS = 5f

        @Volatile
        private var cached: OnlineLearner? = null

        fun get(context: Context, hidden: Int): OnlineLearner =
            cached ?: synchronized(this) {
                cached ?: OnlineLearner(
                    context.applicationContext
                        .getSharedPreferences(PREFS, Context.MODE_PRIVATE),
                    hidden,
                ).also { cached = it }
            }
    }
}
