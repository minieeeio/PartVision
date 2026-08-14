import Constants from 'expo-constants';

const GITHUB_CONFIG_URL = 'https://raw.githubusercontent.com/PrageshShrestha/hasslefree/main/partVision';
const WS_PATH = '/ws/segment';
const FETCH_TIMEOUT_MS = 10000;
const MAX_RETRIES = 3;

export const BACKEND_URL: string =
  Constants?.manifest?.extra?.backendUrl ||
  Constants?.expoConfig?.extra?.backendUrl ||
  '';

export const WS_FRAME_INTERVAL_MS = 200;
export const MAX_RECONNECT_ATTEMPTS = 10;
export const RECONNECT_BASE_DELAY_MS = 500;

export interface BackendConfig {
  api_base_url: string;
}

async function fetchConfigWithRetry(attempt = 1): Promise<BackendConfig> {
  const timeoutPromise = new Promise<never>((_, reject) => {
    setTimeout(() => reject(new Error('Request timed out')), FETCH_TIMEOUT_MS);
  });

  const fetchPromise = (async (): Promise<BackendConfig> => {
    console.log(`[Config] Fetching backend config from ${GITHUB_CONFIG_URL}... (attempt ${attempt}/${MAX_RETRIES})`);
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
    return config;
  })();

  try {
    return await Promise.race([fetchPromise, timeoutPromise]);
  } catch (err) {
    if (attempt < MAX_RETRIES) {
      console.warn(`[Config] Retrying... (${attempt}/${MAX_RETRIES})`);
      await new Promise(r => setTimeout(r, 1000 * attempt));
      return fetchConfigWithRetry(attempt + 1);
    }
    throw err;
  }
}

export async function fetchBackendUrl(): Promise<string> {
  if (BACKEND_URL) {
    return BACKEND_URL;
  }

  try {
    const config = await fetchConfigWithRetry();
    const host = config.api_base_url.replace(/^https?:\/\//, '').replace(/\/$/, '');
    const wsUrl = `wss://${host}${WS_PATH}`;
    console.log(`[Config] Resolved WebSocket URL: ${wsUrl}`);
    return wsUrl;
  } catch (err) {
    console.error('[Config] Failed to resolve backend URL after retries:', err);
    throw new Error('Cannot connect: backend URL not configured and remote config is unreachable.');
  }
}
