class Detection {
  final String label;
  final double confidence;
  final double xMin;
  final double yMin;
  final double width;
  final double height;
  final List<PolygonPoint> polygon;

  Detection({
    required this.label,
    required this.confidence,
    required this.xMin,
    required this.yMin,
    required this.width,
    required this.height,
    required this.polygon,
  });

  factory Detection.fromJson(Map<String, dynamic> json) {
    final polygonList = json['polygon'] as List<dynamic>? ?? [];
    return Detection(
      label: json['label'] as String? ?? 'UNKNOWN',
      confidence: (json['confidence'] as num?)?.toDouble() ?? 0.0,
      xMin: (json['x_min'] as num?)?.toDouble() ?? 0.0,
      yMin: (json['y_min'] as num?)?.toDouble() ?? 0.0,
      width: (json['width'] as num?)?.toDouble() ?? 0.0,
      height: (json['height'] as num?)?.toDouble() ?? 0.0,
      polygon: polygonList
          .map((p) => PolygonPoint(
                x: (p['x'] as num?)?.toDouble() ?? 0.0,
                y: (p['y'] as num?)?.toDouble() ?? 0.0,
              ))
          .toList(),
    );
  }
}

class PolygonPoint {
  final double x;
  final double y;
  const PolygonPoint({required this.x, required this.y});
}
