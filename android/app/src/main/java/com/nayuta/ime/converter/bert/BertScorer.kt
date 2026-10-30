package com.nayuta.ime.converter.bert

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * 文字レベル BERT 採点器の端末推論。
 *
 * 入力は「文脈 + 読み」と、読み中のスパン（文節/単語の範囲）とその候補表記。
 * 出力はスパンごと・候補ごとの log P。
 *
 * モデルは train/export_onnx.py が書き出した int8 量子化済み ONNX。
 */
class BertScorer(context: Context) {

    data class SpanRequest(
        val start: Int,
        val end: Int,
        val surfaces: List<String>,
    )

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val vocab: CharVocab = CharVocab.get(context)
    private val pad: Long = 0L

    init {
        val bytes = context.assets.open(ASSET).use { it.readBytes() }
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(2)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        session = env.createSession(bytes, opts)
    }

    /**
     * @param context 直前の確定済みテキスト（末尾 64 文字を使う）
     * @param reading 変換対象の読み（先頭 48 文字を使う）
     * @return W x K の log P。候補が無い位置は -1e4。
     */
    fun score(
        context: String,
        reading: String,
        spans: List<SpanRequest>,
        k: Int,
    ): Array<FloatArray> {
        if (spans.isEmpty()) return emptyArray()
        val ctx = context.takeLast(MAX_CTX)
        val read = reading.take(MAX_READ)
        val ctxIds = vocab.encode(ctx)
        val readIds = vocab.encode(read)

        val seq = LongArray(1 + ctxIds.size + 1 + readIds.size + 1)
        var p = 0
        seq[p++] = vocab.cls
        ctxIds.copyInto(seq, p); p += ctxIds.size
        seq[p++] = vocab.sep
        readIds.copyInto(seq, p); p += readIds.size
        seq[p] = vocab.sep
        val offset = 1 + ctxIds.size + 1

        val w = spans.size
        val starts = LongArray(w)
        val ends = LongArray(w)
        val candIds = LongArray(w * k * MAX_SURF)
        val candMask = LongArray(w * k * MAX_SURF)
        for (i in spans.indices) {
            val sp = spans[i]
            starts[i] = (offset + sp.start).toLong()
            ends[i] = (offset + sp.end).toLong()
            for (j in 0 until k) {
                if (j >= sp.surfaces.size) continue
                val surf = sp.surfaces[j]
                val ids = vocab.encode(surf)
                val n = minOf(ids.size, MAX_SURF)
                val base = (i * k + j) * MAX_SURF
                for (t in 0 until n) {
                    candIds[base + t] = ids[t]
                    candMask[base + t] = 1L
                }
            }
        }

        val inputs = HashMap<String, OnnxTensor>()
        try {
            inputs["input_ids"] = tensor2(seq, 1, seq.size)
            inputs["attention_mask"] = OnnxTensor.createTensor(
                env, LongBuffer.wrap(LongArray(seq.size) { 1L }), longArrayOf(1, seq.size.toLong()),
            )
            inputs["starts"] = tensor2(starts, 1, w)
            inputs["ends"] = tensor2(ends, 1, w)
            inputs["cand_ids"] = OnnxTensor.createTensor(
                env, LongBuffer.wrap(candIds), longArrayOf(1, w.toLong(), k.toLong(), MAX_SURF.toLong()),
            )
            inputs["cand_mask"] = OnnxTensor.createTensor(
                env, LongBuffer.wrap(candMask), longArrayOf(1, w.toLong(), k.toLong(), MAX_SURF.toLong()),
            )

            session.run(inputs).use { out ->
                @Suppress("UNCHECKED_CAST")
                val logits = out.get(0).value as Array<Array<FloatArray>>
                val res = Array(w) { FloatArray(k) }
                val row = logits[0]
                for (i in 0 until w) {
                    val lr = row[i]
                    var maxv = -1e9f
                    val n = spans[i].surfaces.size
                    for (j in 0 until n) if (lr[j] > maxv) maxv = lr[j]
                    var sum = 0.0
                    for (j in 0 until n) sum += Math.exp((lr[j] - maxv).toDouble())
                    val lse = maxv + Math.log(sum).toFloat()
                    for (j in 0 until k) {
                        res[i][j] = if (j < n) lr[j] - lse else -1e4f
                    }
                }
                return res
            }
        } finally {
            for (t in inputs.values) t.close()
        }
    }

    private fun tensor2(data: LongArray, r: Int, c: Int): OnnxTensor =
        OnnxTensor.createTensor(env, LongBuffer.wrap(data), longArrayOf(r.toLong(), c.toLong()))

    fun close() {
        session.close()
    }

    companion object {
        private const val ASSET = "model/scorer.onnx"
        const val MAX_CTX = 64
        const val MAX_READ = 48
        const val MAX_SURF = 12
        const val MAX_CANDIDATES = 8
    }
}
