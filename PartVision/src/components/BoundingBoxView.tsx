import React from 'react';
import { View, Text, StyleSheet, LayoutRectangle } from 'react-native';
import { PartDetection } from '../types/detection';

interface Props {
  detection: PartDetection;
  containerSize: LayoutRectangle;
}

export const BoundingBoxView: React.FC<Props> = ({ detection, containerSize }) => {
  // Convert 0.0 - 1.0 ratios to screen pixels
  const rect = {
    left: detection.x_min * containerSize.width,
    top: detection.y_min * containerSize.height,
    width: detection.width * containerSize.width,
    height: detection.height * containerSize.height,
  };

  const isBumper = detection.label.includes('BUMPER');
  const accentColor = isBumper ? '#00FFFF' : '#FFFF00';

  return (
    <View
      style={[
        styles.box,
        {
          left: rect.left,
          top: rect.top,
          width: rect.width,
          height: rect.height,
          borderColor: accentColor,
        },
      ]}
    >
      <View style={[styles.labelBadge, { backgroundColor: accentColor }]}>
        <Text style={styles.labelText}>
          {detection.label} ({Math.round(detection.confidence * 100)}%)
        </Text>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  box: {
    position: 'absolute',
    borderWidth: 2,
    backgroundColor: 'rgba(0, 255, 255, 0.05)',
  },
  labelBadge: {
    position: 'absolute',
    top: -20,
    left: -2,
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