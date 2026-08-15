import axios from 'axios';
import { AppConfig } from '../types';

const REMOTE_CONFIG_URL =
  'https://raw.githubusercontent.com/PrageshShrestha/hasslefree/main/partVision.json';

let cachedConfig: AppConfig | null = null;

export async function fetchRemoteConfig(): Promise<AppConfig> {
  if (cachedConfig) {
    return cachedConfig;
  }

  try {
    const response = await axios.get<AppConfig>(REMOTE_CONFIG_URL, {
      timeout: 5000,
    });
    cachedConfig = response.data;
    return cachedConfig;
  } catch (error) {
    throw new Error(`Failed to fetch remote config: ${error}`);
  }
}

export function resolveWebSocketUrl(apiBaseUrl: string): string {
  const trimmed = apiBaseUrl.replace(/\/$/, '');
  if (trimmed.startsWith('https://')) {
    return `${trimmed.replace(/^https/, 'wss')}/ws/segment`;
  }
  if (trimmed.startsWith('http://')) {
    return `${trimmed.replace(/^http/, 'ws')}/ws/segment`;
  }
  return `ws://${trimmed}/ws/segment`;
}

export function getApiBaseUrl(): string {
  // Priority: app.json extra > environment > remote config fallback
  const extra = require('../../app.json').expo?.extra;
  if (extra?.backendUrl) {
    const url = extra.backendUrl;
    if (url.startsWith('ws://') || url.startsWith('wss://')) {
      return url.replace(/^ws/, 'http').replace(/\/ws\/segment$/, '');
    }
    return url.replace(/\/ws\/segment$/, '');
  }
  return '';
}
