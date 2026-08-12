import { BACKEND_URL, GITHUB_CONFIG_URL, CONFIG_CONFIG } from '../utils/constants';

export interface BackendConfig {
  websocket_url: string;
  api_url?: string;
  version?: string;
}

export class ConfigManager {
  private cachedUrl: string | null = null;
  private lastFetch: number = 0;

  async getBackendUrl(): Promise<string> {
    const now = Date.now();

    if (this.cachedUrl && now - this.lastFetch < CONFIG_CONFIG.REFRESH_INTERVAL_MS) {
      return this.cachedUrl;
    }

    try {
      const response = await fetch(GITHUB_CONFIG_URL, {
        method: 'GET',
        headers: { 'Cache-Control': 'no-cache' },
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data: BackendConfig = await response.json();

      if (data.websocket_url) {
        this.cachedUrl = data.websocket_url;
        this.lastFetch = now;
        return data.websocket_url;
      }

      throw new Error('No websocket_url in config response');
    } catch (e) {
      console.warn('[ConfigManager] Fetch failed, falling back to localhost:', e);
      this.cachedUrl = BACKEND_URL;
      this.lastFetch = now;
      return BACKEND_URL;
    }
  }

  clearCache(): void {
    this.cachedUrl = null;
    this.lastFetch = 0;
  }
}

export const configManager = new ConfigManager();
