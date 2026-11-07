import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

// ---------------------------------------------------------------------------
// リリース署名鍵
//
// 鍵は android/keystore.properties（git 管理外）から読む。CI などファイルを
// 置けない環境では NAYUTA_* 環境変数で上書きできる。優先順位は環境変数 > ファイル。
//   NAYUTA_STORE_FILE / NAYUTA_STORE_PASSWORD / NAYUTA_KEY_ALIAS / NAYUTA_KEY_PASSWORD
//
// どちらも無い場合（新規クローン・CI）は、従来どおり debug 署名で release を
// ビルドする。鍵が揃った時点で自動的に release 署名に切り替わる。
// ---------------------------------------------------------------------------
val keystoreProps = Properties().apply {
    val f = rootProject.file("keystore.properties")
    if (f.isFile) f.inputStream().use { load(it) }
}

val envOrProp: (String, String) -> String? = { envName, key ->
    System.getenv(envName)?.takeIf { it.isNotBlank() }
        ?: keystoreProps.getProperty(key)?.takeIf { it.isNotBlank() }
}

// storeFile は android/ からの相対パスとして解決する（例: ../.cache/nayuta-release.p12）。
val releaseStoreFile = envOrProp("NAYUTA_STORE_FILE", "storeFile")?.let { rootProject.file(it) }
val releaseStorePassword = envOrProp("NAYUTA_STORE_PASSWORD", "storePassword")
val releaseKeyAlias = envOrProp("NAYUTA_KEY_ALIAS", "keyAlias") ?: "nayuta"
val releaseKeyPassword = envOrProp("NAYUTA_KEY_PASSWORD", "keyPassword") ?: releaseStorePassword
val hasReleaseKey = releaseStoreFile?.isFile == true && !releaseStorePassword.isNullOrEmpty()

if (!hasReleaseKey && gradle.startParameter.taskNames.any { it.contains("Release", ignoreCase = true) }) {
    logger.lifecycle(
        "warning: release 署名鍵が未設定のため debug 署名でビルドします。" +
            " android/keystore.properties を用意してください（例: keystore.properties.example）。",
    )
}

android {
    namespace = "com.nayuta.ime"
    compileSdk = 35

    // 鍵が揃っている時だけ release 署名を作る。無ければ debug 署名にフォールバック。
    signingConfigs {
        if (hasReleaseKey) {
            create("release") {
                storeFile = releaseStoreFile
                storePassword = releaseStorePassword
                keyAlias = releaseKeyAlias
                keyPassword = releaseKeyPassword
                // v3 署名も有効化しておく（将来 apksigner rotate で鍵ローテーションする余地）。
                // minSdk 26 なので v1(JAR) は不要。
                enableV3Signing = true
            }
        }
    }
    val releaseSigningConfig = signingConfigs.findByName("release")
        ?: signingConfigs.getByName("debug")

    defaultConfig {
        applicationId = "com.nayuta.ime"
        minSdk = 26
        targetSdk = 35
        versionCode = 8
        versionName = "0.2.7"

        // ONNX Runtime のネイティブライブラリを全 ABI 分入れると APK が膨らむため arm64 に絞る
        ndk {
            // 実機は arm64-v8a のみ。配布 APK を小さく保つため。
            // エミュレータ（x86_64）で試すときは -PnayutaAbi=x86_64 を付ける。
            val requested = (project.findProperty("nayutaAbi") as String?)
                ?.split(",")
                ?.map { it.trim() }
                ?.filter { it.isNotEmpty() }
            abiFilters += requested ?: listOf("arm64-v8a")
        }
    }

    // 辞書は mmap で読みたいので .ndict だけ APK 内で非圧縮にする
    androidResources {
        noCompress += "ndict"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            // 鍵が無い環境では debug 署名（従来の挙動）でビルドを通す。
            signingConfig = releaseSigningConfig
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        buildConfig = false
        resValues = false
    }

    lint {
        // 既存コードに NewApi などの lint エラーがあるため、レポートは出すがビルドは失敗させない
        abortOnError = false
    }
}

dependencies {
    implementation(kotlin("stdlib"))
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.31.1")

    // 設定画面を Material Design で組むための依存。
    // IME 本体（キーボード描画）は framework + 自前 View のままで、こちらには依存しない。
    implementation("com.google.android.material:material:1.12.0")
}
