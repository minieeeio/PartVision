package com.react_vision

import android.os.Bundle
import com.facebook.react.ReactActivity
import com.facebook.react.ReactActivityDelegate
import com.facebook.react.defaults.DefaultNewArchitectureEntryPoint
import com.facebook.react.defaults.DefaultReactActivityDelegate

class MainActivity : ReactActivity() {

    override fun getReactNativeHost(): com.facebook.react.ReactNativeHost {
        return (application as com.react_vision.MainApplication).reactNativeHost
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(null)
    }

    override fun getReactActivityDelegate(): ReactActivityDelegate {
        return DefaultReactActivityDelegate(this, mainComponentName, DefaultNewArchitectureEntryPoint.getFabricEnabled())
    }

    override fun getMainComponentName(): String {
        return "react_vision"
    }
}
