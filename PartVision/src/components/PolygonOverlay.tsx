import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import Svg, { Polygon as SvgPolygon } from 'react-native-svg';
import { PartDetection } from '../types/detection';

interface PolygonOverlayProps {
  detections: PartDetection[];
  containerSize: { width: number; height: number };
}

export const PolygonOverlay: React.FC<PolygonOverlayProps> = ({
  detections,
  containerSize,
}) => {
  if (!containerSize.width || !containerSize.height) {
    return null;
  }

  return (
    <View style={[StyleSheet.absoluteFill, styles.container]}>
      <Svg style={StyleSheet.absoluteFill}>
        {detections.map((detection, index) => {
          if (!detection.polygon || detection.polygon.length < 3) {
            return null;
          }

          const points = detection.polygon
            .map((p) => `${p.x * containerSize.width},${p.y * containerSize.height}`)
            .join(' ');

          const isBumper = detection.label.includes('BUMPER');
          const strokeColor = isBumper ? '#00FFFF' : '#FFFF00';
          const fillColor = isBumper ? 'rgba(0,255,255,0.15)' : 'rgba(255,255,0,0.15)';

          return (
            <SvgPolygon
              key={`${detection.label}-${index}`}
              points={points}
              stroke={strokeColor}
              strokeWidth={2}
              fill={fillColor}
            />
          );
        })}
      </Svg>

      {detections.map((detection, index) => {
        if (!detection.polygon || detection.polygon.length < 3) {
          return null;
        }

        const isBumper = detection.label.includes('BUMPER');
        const accentColor = isBumper ? '#00FFFF' : '#FFFF00';

        const left = detection.x_min * containerSize.width;
        const top = detection.y_min * containerSize.height;

        return (
          <View
            key={`label-${detection.label}-${index}`}
            style={[
              styles.labelBadge,
              { left, top, backgroundColor: accentColor },
            ]}
          >
            <Text style={styles.labelText}>
              {detection.label} ({Math.round(detection.confidence * 100)}%)
            </Text>
          </View>
        );
      })}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    pointerEvents: 'none',
  },
  labelBadge: {
    position: 'absolute',
    paddingHorizontal: 4,
    paddingVertical: 2,
  },
  labelText: {
    color: '#000000',
    fontSize: 10,
    fontWeight: 'bold',
    fontFamily: 'monospace',
  },
});
