import React, { useEffect, useState } from 'react';
import { StyleSheet, Text, View, ActivityIndicator, PermissionsAndroid } from 'react-native';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import { Camera } from 'react-native-vision-camera';
import { ScannerScreen } from './src/screens/ScannerScreen';
import { fetchBackendUrl } from './src/config/backend';

export default function App() {
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const [backendUrl, setBackendUrl] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const permissionGranted = await PermissionsAndroid.request(PermissionsAndroid.PERMISSIONS.CAMERA, {
          title: 'Permission Query',
          message: 'PartVision want to access your camera.',
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
      const url = await fetchBackendUrl();
      console.log(`[App] Backend URL resolved: ${url}`);
      setBackendUrl(url);
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

  if (!backendUrl) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color="#00FFFF" />
      </View>
    );
  }

  return (
    <SafeAreaProvider>
      <SafeAreaView style={styles.container}>
        <ScannerScreen backendUrl={backendUrl} />
      </SafeAreaView>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000000' },
  center: { flex: 1, backgroundColor: '#000000', justifyContent: 'center', alignItems: 'center' },
  errorText: { color: '#FF0055', textAlign: 'center', padding: 20 },
});