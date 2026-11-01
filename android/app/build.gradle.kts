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
        versionCode = 4
        versionName = "0.2.3"

        // ONNX Runtime のネイティブライブラリを全 ABI 分入れると APK が膨らむため arm64 に絞る
        ndk {
            // 実機は arm64-v8a のみ。配布 APK を小さく保つため。
            // エミュレータ（x86_64）で試すときは一時的に "x86_64" を足す。
            abiFilters += listOf("arm64-v8a")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
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
}

dependencies {
    implementation(kotlin("stdlib"))
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.31.1")

    // 設定画面を Material Design で組むための依存。
    // IME 本体（キーボード描画）は framework + 自前 View のままで、こちらには依存しない。
    implementation("com.google.android.material:material:1.12.0")
}
