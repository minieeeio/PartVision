import React from 'react';
import { View, StyleSheet, Dimensions } from 'react-native';
import BoundingBoxView from './BoundingBoxView';
import { PartDetection } from '../models/DetectionModel';

export interface BoundingBoxOverlayProps {
  detections: PartDetection[];
}

export default function BoundingBoxOverlay({
  detections,
}: BoundingBoxOverlayProps) {
  const { width, height } = Dimensions.get('window');

  if (!detections.length) return null;

  return (
    <View style={[styles.overlay, { width, height }]} pointerEvents="box-none">
      {detections.map((det, idx) => (
        <BoundingBoxView
          key={`${det.label}_${idx}`}
          detection={det}
          containerWidth={width}
          containerHeight={height}
        />
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  overlay: {
    position: 'absolute',
    top: 0,
    left: 0,
  },
});
