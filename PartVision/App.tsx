import React, { useEffect, useState } from 'react';
import { StyleSheet, Text, View, ActivityIndicator, PermissionsAndroid, Platform } from 'react-native';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import { Camera } from 'react-native-vision-camera';
import { ScannerScreen } from './src/screens/ScannerScreen';
import { fetchBackendUrl } from './src/config/backend';
import * as Location from 'expo-location';

export default function App() {
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const [backendUrl, setBackendUrl] = useState<string | null>(null);
  const [location, setLocation] = useState<Location.LocationObject | null>(null);
  const [locationError, setLocationError] = useState<string | null>(null);
  const [initError, setInitError] = useState<string | null>(null);

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
        } else {
          setHasPermission(false);
        }
      } catch (e) {
        console.warn(e)
        setHasPermission(false);
      }

      let locationGranted = false;
      if (Platform.OS === 'android') {
        const fine = await PermissionsAndroid.request(PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION, {
          title: 'Location Permission',
          message: 'PartVision needs access to your location for GPS tracking.',
          buttonPositive: 'Allow',
        });
        const coarse = await PermissionsAndroid.request(PermissionsAndroid.PERMISSIONS.ACCESS_COARSE_LOCATION, {
          title: 'Location Permission',
          message: 'PartVision needs access to your location for GPS tracking.',
          buttonPositive: 'Allow',
        });
        locationGranted = fine === PermissionsAndroid.RESULTS.GRANTED || coarse === PermissionsAndroid.RESULTS.GRANTED;
      } else {
        const { status } = await Location.requestForegroundPermissionsAsync();
        locationGranted = status === 'granted';
      }

      if (!locationGranted) {
        setLocationError('Location permission denied');
      } else {
        try {
          const subscription = await Location.watchPositionAsync(
            {
              accuracy: Location.Accuracy.High,
              timeInterval: 1000,
              distanceInterval: 1,
            },
            (loc) => {
              setLocation(loc);
            }
          );
          return () => subscription.remove();
        } catch (e) {
          console.warn('[Location] watchPosition failed:', e);
        }
      }

      const resolveBackend = async () => {
        try {
          const url = await fetchBackendUrl();
          console.log(`[App] Backend URL resolved: ${url}`);
          setBackendUrl(url);
          setInitError(null);
        } catch (e) {
          console.warn('[App] Backend URL resolution failed:', e);
          setInitError('Backend URL not configured and remote config is unreachable.');
        }
      };

      await resolveBackend();

      const interval = setInterval(async () => {
        if (!backendUrl) {
          await resolveBackend();
        }
      }, 5000);

      return () => clearInterval(interval);
    })();
  }, []);

  if (hasPermission === false) {
    return (
      <View style={styles.center}>
        <Text style={styles.errorText}>CAMERA PERMISSION IS REQUIRED</Text>
      </View>
    );
  }

  if (locationError) {
    return (
      <View style={styles.center}>
        <Text style={styles.errorText}>{locationError}</Text>
      </View>
    );
  }

  return (
    <SafeAreaProvider>
      <SafeAreaView style={styles.container}>
        <ScannerScreen backendUrl={backendUrl || ''} location={location} />
        {initError && !backendUrl && (
          <View style={styles.errorOverlay}>
            <Text style={styles.errorText}>{initError}</Text>
            <Text style={styles.retryText}>Make sure your phone has internet access.</Text>
          </View>
        )}
      </SafeAreaView>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000000' },
  center: { flex: 1, backgroundColor: '#000000', justifyContent: 'center', alignItems: 'center' },
  errorText: { color: '#FF0055', textAlign: 'center', padding: 20 },
  retryText: { color: '#AAAAAA', textAlign: 'center', padding: 10, fontSize: 12 },
  errorOverlay: {
    position: 'absolute',
    bottom: 40,
    left: 20,
    right: 20,
    backgroundColor: 'rgba(0,0,0,0.85)',
    padding: 16,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#FF0055',
  },
});