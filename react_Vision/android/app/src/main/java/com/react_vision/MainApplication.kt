package com.react_vision

import android.app.Application
import com.facebook.react.ReactApplication
import com.facebook.react.ReactNativeHost
import com.facebook.react.ReactPackage
import com.facebook.react.defaults.DefaultNewArchitectureEntryPoint
import com.facebook.react.defaults.DefaultReactNativeHost
import com.facebook.soloader.SoLoader
import java.util.ArrayList

class MainApplication : Application(), ReactApplication {

    override fun getReactNativeHost(): ReactNativeHost =
        object : DefaultReactNativeHost(this) {
            override fun getPackages(): List<ReactPackage> {
                return listOf(
                    com.facebook.react.shell.MainReactPackage(),
                    com.reactnative.visioncamera.VisionCameraPackage(),
                    com.swmansion.reanimated.ReanimatedPackage(),
                )
            }

            override fun getJSMainModuleName(): String = "index"

            override fun getUseDeveloperSupport(): Boolean = BuildConfig.DEBUG

            override fun getBundleAssetName(): String = "index.android.bundle"

            override fun getJavaScriptBundleFile(): String? {
                return if (BuildConfig.DEBUG) {
                    super.getJavaScriptBundleFile()
                } else {
                    super.getJavaScriptBundleFile()
                }
            }
        }

    override fun onCreate() {
        super.onCreate()
        SoLoader.init(this, /* native exopackage */ false)
        if (BuildConfig.DEBUG) {
            DefaultNewArchitectureEntryPoint.getSWCEnabled()
        }
    }
}
