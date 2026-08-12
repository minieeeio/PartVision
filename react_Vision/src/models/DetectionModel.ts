export interface PartDetection {
  label: string;
  confidence: number;
  x_min: number;
  y_min: number;
  width: number;
  height: number;
}

export interface DetectionResponse {
  detections: PartDetection[];
  process_time_ms?: number;
}

export interface WebSocketError {
  code: string;
  message: string;
  timestamp: number;
}

export interface StreamingStats {
  fps: number;
  bytesSent: number;
  bytesReceived: number;
  totalFrames: number;
  lastProcessTimeMs?: number;
}

export type ConnectionState =
  | 'disconnected'
  | 'connecting'
  | 'connected'
  | 'error';
