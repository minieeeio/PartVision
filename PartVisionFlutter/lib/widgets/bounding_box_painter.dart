import 'package:flutter/material.dart';
import '../models/detection.dart';

class BoundingBoxPainter extends CustomPainter {
  final List<Detection> detections;
  final Size frameSize;
  final Size viewSize;

  BoundingBoxPainter({
    required this.detections,
    required this.frameSize,
    required this.viewSize,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (detections.isEmpty || frameSize.width == 0 || frameSize.height == 0) return;

    final scaleX = size.width / frameSize.width;
    final scaleY = size.height / frameSize.height;
    final scale = scaleX > scaleY ? scaleX : scaleY;

    final visibleWidth = size.width / scale;
    final visibleHeight = size.height / scale;
    final offsetX = (frameSize.width - visibleWidth) / 2;
    final offsetY = (frameSize.height - visibleHeight) / 2;

    double mapX(double nx) => (nx * frameSize.width - offsetX) * scale;
    double mapY(double ny) => (ny * frameSize.height - offsetY) * scale;
    double mapW(double nw) => nw * frameSize.width * scale;
    double mapH(double nh) => nh * frameSize.height * scale;

    final strokePaint = Paint()
      ..color = const Color(0xFF00ff88)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;

    final fillPaint = Paint()
      ..color = const Color(0x2600ff88)
      ..style = PaintingStyle.fill;

    final polygonPaint = Paint()
      ..color = const Color(0x6600ff88)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1;

    for (final det in detections) {
      final x = mapX(det.xMin);
      final y = mapY(det.yMin);
      final w = mapW(det.width);
      final h = mapH(det.height);

      if (x + w < 0 || y + h < 0 || x > size.width || y > size.height) continue;

      final rect = Rect.fromLTWH(x, y, w, h);
      canvas.drawRect(rect, fillPaint);
      canvas.drawRect(rect, strokePaint);

      final labelText = '${det.label} ${(det.confidence * 100).toInt()}%';
      final textPainter = TextPainter(
        text: TextSpan(
          text: labelText,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 12,
            fontWeight: FontWeight.w600,
            fontFamily: '.SF Pro Text',
          ),
        ),
        textDirection: TextDirection.ltr,
      );
      textPainter.layout();

      final labelY = y > 24 ? y - 8 : y + h + 20;
      final labelBgRect = Rect.fromLTWH(
        x,
        labelY - 16,
        textPainter.width + 12,
        18,
      );
      final bgPaint = Paint()..color = const Color(0xB3000000);
      canvas.drawRRect(
        RRect.fromRectAndRadius(labelBgRect, const Radius.circular(4)),
        bgPaint,
      );
      textPainter.paint(canvas, Offset(x + 6, labelY - 14));

      if (det.polygon.isNotEmpty) {
        final path = Path();
        final firstX = mapX(det.polygon.first.x);
        final firstY = mapY(det.polygon.first.y);
        path.moveTo(firstX, firstY);

        for (int i = 1; i < det.polygon.length; i++) {
          path.lineTo(mapX(det.polygon[i].x), mapY(det.polygon[i].y));
        }
        path.close();
        canvas.drawPath(path, polygonPaint);
      }
    }
  }

  @override
  bool shouldRepaint(BoundingBoxPainter oldDelegate) {
    return oldDelegate.detections.length != detections.length ||
        oldDelegate.frameSize != frameSize ||
        oldDelegate.viewSize != viewSize;
  }
}
