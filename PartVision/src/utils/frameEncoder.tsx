import { Frame } from 'react-native-vision-camera';

/**
 * Prepares camera frame data for network transmission.
 * Runs on a dedicated JS worklet thread.
 */
export const encodeFrameToBase64 = (frame: Frame): string | null => {
  'worklet';
  if (!frame || !frame.isValid) return null;

  try {
    // Generates a structured frame payload string
    return JSON.stringify({
      width: frame.width,
      height: frame.height,
      timestamp: frame.timestamp,
    });
  } catch (error) {
    return null;
  }
};