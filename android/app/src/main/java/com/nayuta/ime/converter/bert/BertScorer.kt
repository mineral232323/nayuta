package com.nayuta.ime.converter.bert

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * 文字レベル BERT 採点器の端末推論。オンライン学習用の出力も扱う。
 *
 * モデルの入出力:
 *   inputs : input_ids / attention_mask / starts / ends / cand_ids / cand_mask / w
 *   outputs: logits (B,W,K) / extra (B,W,K) / phi (B,W,K,H)
 *
 * - [score] … logits と extra だけを要求（phi は ONNX ランタイム側で刈られる）
 * - [phiForSpan] … 確定したスパン 1 つ分の phi を要求（W=1 の軽い forward）
 */
class BertScorer(context: Context) {

    data class SpanRequest(
        val start: Int,
        val end: Int,
        val surfaces: List<String>,
    )

    /** 確定時の学習に使う材料。 */
    data class PhiResult(
        val logits: FloatArray,          // (K)
        val extra: FloatArray,           // (K)
        val phi: Array<FloatArray>,      // (K, H)
    )

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val vocab: CharVocab = CharVocab.get(context)

    /** 学習する薄い層の次元（モデルの w 入力から取る）。 */
    val hidden: Int

    init {
        val bytes = context.assets.open(ASSET).use { it.readBytes() }
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(2)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        session = env.createSession(bytes, opts)
        val wInfo = session.inputInfo["w"] as? ai.onnxruntime.TensorInfo
        hidden = wInfo?.shape?.getOrNull(1)?.toInt() ?: 384
    }

    /**
     * スパンごとの最終スコア（logits + extra）を返す。
     * 呼び出し側で softmax して使う。
     */
    fun score(
        context: String,
        reading: String,
        spans: List<SpanRequest>,
        k: Int,
        w: FloatArray,
    ): Array<FloatArray> {
        if (spans.isEmpty()) return emptyArray()
        val inputs = buildInputs(context, reading, spans, k, w)
        try {
            session.run(inputs, setOf("logits", "extra")).use { out ->
                @Suppress("UNCHECKED_CAST")
                val logits = out.get("logits").get().value as Array<Array<FloatArray>>
                @Suppress("UNCHECKED_CAST")
                val extra = out.get("extra").get().value as Array<Array<FloatArray>>
                val wCount = spans.size
                val res = Array(wCount) { FloatArray(k) }
                for (i in 0 until wCount) {
                    val n = spans[i].surfaces.size
                    for (j in 0 until k) {
                        res[i][j] = if (j < n) logits[0][i][j] + extra[0][i][j] else -1e4f
                    }
                }
                return res
            }
        } finally {
            inputs.values.forEach { it.close() }
        }
    }

    /** 確定した 1 スパン分の logits / extra / phi を取る（学習用）。 */
    fun phiForSpan(
        context: String,
        reading: String,
        span: SpanRequest,
        k: Int,
        w: FloatArray,
    ): PhiResult? {
        val inputs = buildInputs(context, reading, listOf(span), k, w)
        try {
            session.run(inputs, setOf("logits", "extra", "phi")).use { out ->
                @Suppress("UNCHECKED_CAST")
                val logits = out.get("logits").get().value as Array<Array<FloatArray>>
                @Suppress("UNCHECKED_CAST")
                val extra = out.get("extra").get().value as Array<Array<FloatArray>>
                @Suppress("UNCHECKED_CAST")
                val phi = out.get("phi").get().value as Array<Array<Array<FloatArray>>>
                val n = span.surfaces.size
                if (n == 0) return null
                val l = FloatArray(n) { logits[0][0][it] }
                val e = FloatArray(n) { extra[0][0][it] }
                val p = Array(n) { c -> FloatArray(hidden) { d -> phi[0][0][c][d] } }
                return PhiResult(l, e, p)
            }
        } finally {
            inputs.values.forEach { it.close() }
        }
    }

    // ------------------------------------------------------------------

    private fun buildInputs(
        context: String,
        reading: String,
        spans: List<SpanRequest>,
        k: Int,
        w: FloatArray,
    ): HashMap<String, OnnxTensor> {
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

        val wCount = spans.size
        val starts = LongArray(wCount)
        val ends = LongArray(wCount)
        val candIds = LongArray(wCount * k * MAX_SURF)
        val candMask = LongArray(wCount * k * MAX_SURF)
        for (i in spans.indices) {
            val sp = spans[i]
            starts[i] = (offset + sp.start).toLong()
            ends[i] = (offset + sp.end).toLong()
            for (j in 0 until k) {
                if (j >= sp.surfaces.size) continue
                val ids = vocab.encode(sp.surfaces[j])
                val n = minOf(ids.size, MAX_SURF)
                val base = (i * k + j) * MAX_SURF
                for (t in 0 until n) {
                    candIds[base + t] = ids[t]
                    candMask[base + t] = 1L
                }
            }
        }

        val wBuf = FloatArray(hidden)
        w.copyInto(wBuf, 0, 0, minOf(hidden, w.size))

        return HashMap<String, OnnxTensor>().apply {
            put("input_ids", OnnxTensor.createTensor(env, LongBuffer.wrap(seq), longArrayOf(1, seq.size.toLong())))
            put(
                "attention_mask",
                OnnxTensor.createTensor(
                    env,
                    LongBuffer.wrap(LongArray(seq.size) { 1L }),
                    longArrayOf(1, seq.size.toLong()),
                ),
            )
            put("starts", OnnxTensor.createTensor(env, LongBuffer.wrap(starts), longArrayOf(1, wCount.toLong())))
            put("ends", OnnxTensor.createTensor(env, LongBuffer.wrap(ends), longArrayOf(1, wCount.toLong())))
            put(
                "cand_ids",
                OnnxTensor.createTensor(
                    env,
                    LongBuffer.wrap(candIds),
                    longArrayOf(1, wCount.toLong(), k.toLong(), MAX_SURF.toLong()),
                ),
            )
            put(
                "cand_mask",
                OnnxTensor.createTensor(
                    env,
                    LongBuffer.wrap(candMask),
                    longArrayOf(1, wCount.toLong(), k.toLong(), MAX_SURF.toLong()),
                ),
            )
            put(
                "w",
                OnnxTensor.createTensor(env, FloatBuffer.wrap(wBuf), longArrayOf(1, hidden.toLong())),
            )
        }
    }

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
