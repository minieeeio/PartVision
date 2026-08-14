import Constants from 'expo-constants';

const DEFAULT_BACKEND_URL = 'ws://192.168.1.100:5555/ws/segment';
const GITHUB_CONFIG_URL = 'https://raw.githubusercontent.com/PrageshShrestha/hasslefree/main/partVision';
const WS_PATH = '/ws/segment';

export const BACKEND_URL: string =
  Constants?.manifest?.extra?.backendUrl ||
  Constants?.expoConfig?.extra?.backendUrl ||
  DEFAULT_BACKEND_URL;

export const WS_FRAME_INTERVAL_MS = 200;
export const MAX_RECONNECT_ATTEMPTS = 10;
export const RECONNECT_BASE_DELAY_MS = 500;

export interface BackendConfig {
  api_base_url: string;
}

export async function fetchBackendUrl(): Promise<string> {
  try {
    console.log(`[Config] Fetching backend config from ${GITHUB_CONFIG_URL}...`);
    const response = await fetch(GITHUB_CONFIG_URL, {
      headers: { Accept: 'application/json' },
    });
    if (!response.ok) {
      throw new Error(`GitHub config fetch failed: ${response.status}`);
    }
    const config: BackendConfig = await response.json();
    console.log(`[Config] Fetched config:`, config);
    if (!config?.api_base_url) {
      throw new Error('Missing api_base_url in config');
    }
    const host = config.api_base_url.replace(/^https?:\/\//, '').replace(/\/$/, '');
    const wsUrl = `wss://${host}${WS_PATH}`;
    console.log(`[Config] Resolved WebSocket URL: ${wsUrl}`);
    return wsUrl;
  } catch (err) {
    console.warn('[Config] Falling back to default backend URL:', err);
    return DEFAULT_BACKEND_URL;
  }
}
