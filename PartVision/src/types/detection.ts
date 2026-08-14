// Represents a single detected object returned by the AI model
export interface PartDetection {
  label: string;      // e.g., "FRONT_BUMPER", "HEADLIGHT_L"
  confidence: number; // e.g., 0.92 (92% certainty)
  x_min: number;      // Normalized 0.0 - 1.0 bounding box coordinates
  y_min: number;
  width: number;
  height: number;
  polygon?: { x: number; y: number }[]; // Normalized polygon points from segmentation mask
}

// The full JSON response wrapper received over WebSocket from FastAPI
export interface DetectionResponse {
  detections: PartDetection[];
  process_time_ms: number; // Backend latency measurement
}