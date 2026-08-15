export interface Detection {
  label: string;
  confidence: number;
  x_min: number;
  y_min: number;
  width: number;
  height: number;
  polygon: { x: number; y: number }[];
}

export interface DetectionResponse {
  detections: Detection[];
  process_time_ms: number;
  location?: { latitude: number; longitude: number; accuracy?: number };
}

export interface AppConfig {
  api_base_url: string;
}

export type CameraPermissionStatus = 'not-determined' | 'denied' | 'authorized';
