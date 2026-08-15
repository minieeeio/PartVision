import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'package:camera/camera.dart';

class FrameProcessor {
  static const int targetFps = 1;
  static const int frameIntervalMs = 1000 ~/ targetFps;

  static Future<Uint8List?> encodeJpeg(CameraImage image) async {
    try {
      final rgb = _convertYUV420ToImage(image);
      if (rgb == null) {
        print('[Frame] YUV conversion returned null');
        return null;
      }
      final jpg = img.encodeJpg(rgb, quality: 70);
      print('[Frame] Encoded JPEG: ${jpg.length} bytes');
      return Uint8List.fromList(jpg);
    } catch (e) {
      print('[Frame] Encoding error: $e');
      return null;
    }
  }

  static img.Image? _convertYUV420ToImage(CameraImage image) {
    final width = image.width;
    final height = image.height;
    final yuv = image.planes;

    if (yuv.length < 3) return null;

    final yPlane = yuv[0].bytes;
    final uPlane = yuv[1].bytes;
    final vPlane = yuv[2].bytes;

    final rgb = img.Image(width: width, height: height);

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final yIndex = y * width + x;
        final yValue = yPlane[yIndex];

        final uvRow = (y ~/ 2) * (width ~/ 2);
        final uvCol = x ~/ 2;
        final uvOffset = uvRow + uvCol;

        final uValue = uPlane[uvOffset];
        final vValue = vPlane[uvOffset];

        int r = (yValue + 1.402 * (vValue - 128)).toInt();
        int g = (yValue - 0.344 * (uValue - 128) - 0.714 * (vValue - 128)).toInt();
        int b = (yValue + 1.772 * (uValue - 128)).toInt();

        r = r.clamp(0, 255);
        g = g.clamp(0, 255);
        b = b.clamp(0, 255);

        rgb.setPixelRgb(x, y, r, g, b);
      }
    }

    return rgb;
  }
}
