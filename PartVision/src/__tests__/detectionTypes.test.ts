import { PartDetection, DetectionResponse } from '../types/detection';

describe('Detection Types', () => {
  it('PartDetection has all required fields', () => {
    const detection: PartDetection = {
      label: 'FRONT_BUMPER',
      confidence: 0.92,
      x_min: 0.1,
      y_min: 0.2,
      width: 0.5,
      height: 0.3,
    };
    expect(detection.label).toBe('FRONT_BUMPER');
    expect(detection.confidence).toBeCloseTo(0.92);
    expect(detection.x_min).toBe(0.1);
    expect(detection.y_min).toBe(0.2);
    expect(detection.width).toBe(0.5);
    expect(detection.height).toBe(0.3);
  });

  it('DetectionResponse wraps detections and process_time_ms', () => {
    const response: DetectionResponse = {
      detections: [],
      process_time_ms: 45.23,
    };
    expect(Array.isArray(response.detections)).toBe(true);
    expect(response.process_time_ms).toBe(45.23);
  });

  it('supports all 24 car part labels from the ONNX model', () => {
    const labels = [
      'BACKGROUND',
      'BACK_BUMPER', 'BACK_DOOR', 'BACK_GLASS', 'BACK_LEFT_DOOR',
      'BACK_LEFT_LIGHT', 'BACK_LIGHT', 'BACK_RIGHT_DOOR', 'BACK_RIGHT_LIGHT',
      'FRONT_BUMPER', 'FRONT_DOOR', 'FRONT_GLASS', 'FRONT_LEFT_DOOR',
      'FRONT_LEFT_LIGHT', 'FRONT_LIGHT', 'FRONT_RIGHT_DOOR', 'FRONT_RIGHT_LIGHT',
      'HOOD', 'LEFT_MIRROR', 'OBJECT', 'RIGHT_MIRROR',
      'TAILGATE', 'TRUNK', 'WHEEL',
    ];
    expect(labels.length).toBe(24);
    labels.forEach(label => {
      const detection: PartDetection = {
        label,
        confidence: 0.85,
        x_min: 0,
        y_min: 0,
        width: 0,
        height: 0,
      };
      expect(detection.label).toBe(label);
    });
  });

  it('bounding box label includes BUMPER for bumper parts', () => {
    const detection: PartDetection = {
      label: 'FRONT_BUMPER',
      confidence: 0.90,
      x_min: 0, y_min: 0, width: 0, height: 0,
    };
    expect(detection.label.includes('BUMPER')).toBe(true);
  });

  it('bounding box label does not include BUMPER for non-bumper parts', () => {
    const detection: PartDetection = {
      label: 'HOOD',
      confidence: 0.90,
      x_min: 0, y_min: 0, width: 0, height: 0,
    };
    expect(detection.label.includes('BUMPER')).toBe(false);
  });
});