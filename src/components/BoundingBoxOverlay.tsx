import React from 'react';
import { View, StyleSheet, useWindowDimensions } from 'react-native';
import Svg, { Rect, G, Text as SvgText, Line } from 'react-native-svg';
import { Detection } from '../types';

interface BoundingBoxOverlayProps {
  detections: Detection[];
  frameWidth: number;
  frameHeight: number;
}

export default function BoundingBoxOverlay({
  detections,
  frameWidth,
  frameHeight,
}: BoundingBoxOverlayProps) {
  const { width: containerWidth, height: containerHeight } = useWindowDimensions();

  if (detections.length === 0 || frameWidth === 0 || frameHeight === 0) {
    return null;
  }

  const scaleX = containerWidth / frameWidth;
  const scaleY = containerHeight / frameHeight;
  const scale = Math.max(scaleX, scaleY);

  const visibleWidth = containerWidth / scale;
  const visibleHeight = containerHeight / scale;
  const offsetX = (frameWidth - visibleWidth) / 2;
  const offsetY = (frameHeight - visibleHeight) / 2;

  const mapX = (nx: number) => (nx * frameWidth - offsetX) * scale;
  const mapY = (ny: number) => (ny * frameHeight - offsetY) * scale;
  const mapW = (nw: number) => nw * frameWidth * scale;
  const mapH = (nh: number) => nh * frameHeight * scale;

  const mappedDetections = detections.map((det) => ({
    ...det,
    x: mapX(det.x_min),
    y: mapY(det.y_min),
    w: mapW(det.width),
    h: mapH(det.height),
  }));

  const strokeColor = '#00ff88';
  const fillColor = 'rgba(0, 255, 136, 0.15)';
  const labelBgColor = 'rgba(0, 0, 0, 0.75)';
  const labelTextColor = '#ffffff';

  return (
    <View style={StyleSheet.absoluteFill} pointerEvents="none">
      <Svg width={containerWidth} height={containerHeight}>
        {mappedDetections.map((det, index) => {
          const isInside =
            det.x >= -det.w &&
            det.y >= -det.h &&
            det.x + det.w <= containerWidth + det.w &&
            det.y + det.h <= containerHeight + det.h;

          if (!isInside) return null;

          const labelY = det.y > 24 ? det.y - 8 : det.y + det.h + 20;
          const labelText = `${det.label} ${(det.confidence * 100).toFixed(0)}%`;

          return (
            <G key={`${det.label}-${index}`}>
              <Rect
                x={det.x}
                y={det.y}
                width={det.w}
                height={det.h}
                stroke={strokeColor}
                strokeWidth={2}
                fill={fillColor}
              />
              <Rect
                x={det.x}
                y={labelY - 16}
                width={labelText.length * 8 + 12}
                height={18}
                fill={labelBgColor}
                rx={4}
              />
              <SvgText
                x={det.x + 6}
                y={labelY - 2}
                fontSize={12}
                fontWeight="600"
                fill={labelTextColor}
                fontFamily="system-ui"
              >
                {labelText}
              </SvgText>
              {det.polygon && det.polygon.length > 0 && (
                <G opacity={0.4}>
                  {det.polygon.map((pt, i) => {
                    const px = mapX(pt.x);
                    const py = mapY(pt.y);
                    const next = det.polygon![(i + 1) % det.polygon!.length];
                    const nextPx = mapX(next.x);
                    const nextPy = mapY(next.y);
                    return (
                      <Line
                        key={`poly-${index}-${i}`}
                        x1={px}
                        y1={py}
                        x2={nextPx}
                        y2={nextPy}
                        stroke={strokeColor}
                        strokeWidth={1}
                        strokeDasharray={[4, 4]}
                      />
                    );
                  })}
                </G>
              )}
            </G>
          );
        })}
      </Svg>
    </View>
  );
}
