package com.nayuta.ime.keyboard

import android.graphics.Color

/**
 * Gboard 風 / Material 3（Material You）ライトテーマの配色トークン。
 *
 * Material Design のカラーロールに合わせて命名している:
 * - surface / surfaceContainer: キーボード地・候補ストリップ
 * - surfaceContainerLowest: キー（白）
 * - onSurface / onSurfaceVariant: 主ラベル / サブラベル
 * - primary: 選択中のサブかな・状態レイヤーの色味
 * - primaryContainer / onPrimaryContainer: 選択中の候補ピル
 *
 * 押下時は「primary を 12% 載せた状態レイヤー」を白キーの上に合成した色を
 * [keyPressed] として事前計算している（Gboard の押下表現に近い）。
 */
object KeyboardColors {

    // ---- surface ----
    /** キーボード全体の地。 */
    val keyboardBackground: Int = Color.parseColor("#F1F3F4")

    /** 候補（サジェスチョン）ストリップの地。キーボードより一段明るい。 */
    val stripBackground: Int = Color.parseColor("#F8F9FA")

    /** 通常キー。 */
    val key: Int = Color.parseColor("#FFFFFF")

    /** 通常キーの押下（primary 12% state layer 合成後）。 */
    val keyPressed: Int = Color.parseColor("#E8F0FE")

    /** 機能キー（英数 / 空白 / 変換 / 削除）。面の色で区別する。 */
    val functionKey: Int = Color.parseColor("#E6E8EB")

    /** 機能キーの押下。 */
    val functionKeyPressed: Int = Color.parseColor("#D3D7DC")

    // ---- on-surface ----
    /** キーの主ラベル。 */
    val keyText: Int = Color.parseColor("#1B1C1E")

    /** フリックのサブラベル。 */
    val keySubText: Int = Color.parseColor("#5F6368")

    /** 押下中に選ばれているサブラベル（primary）。 */
    val keySubTextActive: Int = Color.parseColor("#0B57D0")

    // ---- popup ----
    val balloonBackground: Int = Color.parseColor("#FFFFFF")
    val balloonText: Int = Color.parseColor("#1B1C1E")
    val balloonOutline: Int = Color.parseColor("#E3E6EA")
    val balloonShadow: Int = Color.argb(7, 0, 0, 0)

    // ---- candidates ----
    val candidateText: Int = Color.parseColor("#1B1C1E")
    val candidateSelectedText: Int = Color.parseColor("#041E49")
    val candidateSelectedBackground: Int = Color.parseColor("#D3E3FD")

    /** ストリップとキーボードの境界線（hairline）。 */
    val divider: Int = Color.parseColor("#E3E6EA")

    /** リップル（タップ状態レイヤー）。 */
    val ripple: Int = Color.parseColor("#1F000000")
}
