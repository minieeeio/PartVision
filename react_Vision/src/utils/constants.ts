export const BACKEND_URL = 'ws://localhost:8000/ws/segment';

export const GITHUB_CONFIG_URL =
  'https://raw.githubusercontent.com/PartVision/config/main/backend.json';

export const CAMERA_CONFIG = {
  TARGET_FRAME_WIDTH: 640,
  JPEG_QUALITY: 0.5,
  MAX_FPS: 15,
};

export const WEBSOCKET_CONFIG = {
  RECONNECT_INTERVAL_MS: 2000,
  MAX_RECONNECT_ATTEMPTS: 10,
  HEARTBEAT_INTERVAL_MS: 15000,
  FRAME_DEBOUNCE_MS: 66,
};

export const CONFIG_CONFIG = {
  REFRESH_INTERVAL_MS: 5 * 60 * 1000,
};
