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
        versionCode = 1
        versionName = "0.2.0"

        // ONNX Runtime のネイティブライブラリを全 ABI 分入れると APK が膨らむため arm64 に絞る
        ndk {
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
    // The IME intentionally depends only on the Android framework + Kotlin stdlib,
    // so this project builds without any AndroidX artifacts.
    implementation(kotlin("stdlib"))
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.31.1")
}
