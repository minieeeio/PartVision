import React from 'react';
import { View, StyleSheet, Text } from 'react-native';
import { PartDetection } from '../models/DetectionModel';

export interface BoundingBoxViewProps {
  detection: PartDetection;
  containerWidth: number;
  containerHeight: number;
}

export default function BoundingBoxView({
  detection,
  containerWidth,
  containerHeight,
}: BoundingBoxViewProps) {
  const { x_min, y_min, width, height, label, confidence } = detection;

  const rect = {
    x: x_min * containerWidth,
    y: y_min * containerHeight,
    w: width * containerWidth,
    h: height * containerHeight,
  };

  const isBumperOrLight =
    label.includes('BUMPER') || label.includes('LIGHT');
  const boxColor = isBumperOrLight ? '#00ffff' : '#ffff00';

  if (rect.w < 4 || rect.h < 4) return null;

  const confText = `${Math.round(confidence * 100)}%`;

  return (
    <View
      style={[
        styles.boxContainer,
        {
          left: rect.x,
          top: rect.y,
          width: rect.w,
          height: rect.h,
        },
      ]}
      pointerEvents="none"
    >
      <View
        style={[
          styles.boxOutline,
          {
            width: rect.w,
            height: rect.h,
            borderColor: boxColor,
          },
        ]}
      />
      <View
        style={[
          styles.labelBadge,
          { backgroundColor: boxColor },
        ]}
      >
        <Text style={styles.labelText}>{label}</Text>
      </View>
      <View
        style={[
          styles.confBadge,
          { backgroundColor: 'rgba(0, 0, 0, 0.75)' },
        ]}
      >
        <Text style={[styles.confText, { color: boxColor }]}>{confText}</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  boxContainer: {
    position: 'absolute',
  },
  boxOutline: {
    position: 'absolute',
    borderWidth: 2,
    borderRadius: 2,
  },
  labelBadge: {
    position: 'absolute',
    top: 0,
    left: 0,
    paddingHorizontal: 6,
    paddingVertical: 3,
    borderRadius: 2,
  },
  labelText: {
    fontSize: 10,
    fontWeight: '700',
    fontFamily: 'monospace',
    color: '#000',
  },
  confBadge: {
    position: 'absolute',
    right: 0,
    bottom: 0,
    paddingHorizontal: 2,
    borderRadius: 2,
  },
  confText: {
    fontSize: 9,
    fontWeight: '700',
    fontFamily: 'monospace',
  },
});
