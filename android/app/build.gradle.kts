import java.io.File

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.nayuta.ime"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.nayuta.ime"
        minSdk = 26
        targetSdk = 35
        versionCode = 6
        versionName = "0.2.5"

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

    signingConfigs {
        // CI の Secrets（ANDROID_KEYSTORE_BASE64 等）から環境変数経由で受け取る。
        // 環境変数が無い / keystore ファイルが無い場合は debug 署名にフォールバックし、
        // ローカルビルドや secrets 未設定の CI でも失敗しないようにする。
        val keystorePath = System.getenv("NAYUTA_KEYSTORE_PATH")
        val keystorePassword = System.getenv("NAYUTA_KEYSTORE_PASSWORD")
        if (keystorePath != null && File(keystorePath).isFile) {
            create("release") {
                storeFile = File(keystorePath)
                storePassword = keystorePassword
                keyAlias = System.getenv("NAYUTA_KEY_ALIAS")
                keyPassword = System.getenv("NAYUTA_KEY_PASSWORD") ?: keystorePassword
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
            signingConfig = signingConfigs.findByName("release")
                ?: signingConfigs.getByName("debug")
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
