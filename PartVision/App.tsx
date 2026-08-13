import React, { useEffect, useState } from 'react';
import { StyleSheet, Text, View, ActivityIndicator,PermissionsAndroid } from 'react-native';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
// Explicitly import Camera from react-native-vision-camera ONLY
import { Camera } from 'react-native-vision-camera'; 
import { ScannerScreen } from './src/screens/ScannerScreen';

export default function App() {
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const permissionGranted = await PermissionsAndroid.request(PermissionsAndroid.PERMISSIONS.CAMERA, {
        title: 'Permission Query',
        message:
          'PartVision want to access your camera.',
        buttonNegative: 'Deny',
        buttonPositive: 'Allow',
      });
      if ('granted' === PermissionsAndroid.RESULTS.GRANTED) {
        setHasPermission(true)
    }
      } catch (e) {
        console.warn(e)
        setHasPermission(false);
      }
    })();
  }, []);

  if (hasPermission === null) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color="#00FFFF" />
      </View>
    );
  }

  if (!hasPermission) {
    return (
      <View style={styles.center}>
        <Text style={styles.errorText}>CAMERA PERMISSION IS REQUIRED</Text>
      </View>
    );
  }

  return (
    <SafeAreaProvider>
      <SafeAreaView style={styles.container}>
        <ScannerScreen />
      </SafeAreaView>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000000' },
  center: { flex: 1, backgroundColor: '#000000', justifyContent: 'center', alignItems: 'center' },
  errorText: { color: '#FF0055', textAlign: 'center', padding: 20 },
});