package com.nayuta.ime.converter.dict

import android.content.Context
import android.content.SharedPreferences
import java.util.concurrent.ConcurrentHashMap

/**
 * ユーザー辞書。訂正パネルで登録した (読み, 表記) を端末内に保存する。
 *
 * 変換時はこの辞書が**最優先**で候補に載る（コスト 0）。
 * 「いーろんますく → イーロンマスク」のように、辞書に無い固有名詞を
 * 一度登録すれば次回から先頭に出るようにするためのもの。
 *
 * 保存先は SharedPreferences のみ。外部送信は一切しない。
 */
class UserDictionary private constructor(
    private val prefs: SharedPreferences,
) {

    private val cache = ConcurrentHashMap<String, MutableList<String>>()

    init {
        load()
    }

    private fun load() {
        val raw = prefs.getString(KEY, "").orEmpty()
        if (raw.isEmpty()) return
        for (line in raw.split('\n')) {
            if (line.isEmpty()) continue
            val i = line.indexOf('\t')
            if (i <= 0) continue
            val reading = line.substring(0, i)
            val surface = line.substring(i + 1)
            cache.getOrPut(reading) { mutableListOf() }.add(surface)
        }
    }

    /** 同じ読みに同じ表記を二重登録しない。 */
    fun add(reading: String, surface: String) {
        if (reading.isEmpty() || surface.isEmpty()) return
        val list = cache.getOrPut(reading) { mutableListOf() }
        if (list.contains(surface)) return
        list.add(0, surface)
        persist()
    }

    fun entries(reading: String): List<String> = cache[reading] ?: emptyList()

    fun remove(reading: String, surface: String) {
        val list = cache[reading] ?: return
        list.remove(surface)
        if (list.isEmpty()) cache.remove(reading)
        persist()
    }

    val size: Int get() = cache.values.sumOf { it.size }

    /** 全件削除。 */
    fun clear() {
        cache.clear()
        persist()
    }

    private fun persist() {
        val sb = StringBuilder()
        for ((reading, surfaces) in cache) {
            for (s in surfaces) {
                sb.append(reading).append('\t').append(s).append('\n')
            }
        }
        prefs.edit().putString(KEY, sb.toString()).apply()
    }

    companion object {
        private const val PREFS = "nayuta_user_dict"
        private const val KEY = "entries"

        @Volatile
        private var cached: UserDictionary? = null

        fun get(context: Context): UserDictionary =
            cached ?: synchronized(this) {
                cached ?: UserDictionary(
                    context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE),
                ).also { cached = it }
            }
    }
}
