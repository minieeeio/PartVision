import { Frame, HybridFrameConverter } from 'react-native-vision-camera';

export interface EncodedFrameData {
  buffer: ArrayBuffer;
  width: number;
  height: number;
}

const TARGET_WIDTH = 640;
const TARGET_HEIGHT = 480;
const JPEG_QUALITY = 70;

export const encodeFrameToJpeg = (frame: Frame): EncodedFrameData | null => {
  'worklet';
  if (!frame || !frame.isValid) return null;

  try {
    const image = HybridFrameConverter.convertFrameToImage(frame);
    const resized = image.resize(TARGET_WIDTH, TARGET_HEIGHT);
    const encoded = resized.toEncodedImageData('jpg', JPEG_QUALITY);

    return {
      buffer: encoded.buffer,
      width: encoded.width,
      height: encoded.height,
    };
  } catch {
    return null;
  }
};
