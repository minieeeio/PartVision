import { BACKEND_URL, WS_FRAME_INTERVAL_MS, MAX_RECONNECT_ATTEMPTS, RECONNECT_BASE_DELAY_MS } from '../config/backend';

describe('Backend Config', () => {
  it('defines a valid WebSocket URL', () => {
    expect(BACKEND_URL).toMatch(/^ws[s]?:\/\//);
  });

  it('defines a 200ms frame interval (5 FPS max)', () => {
    expect(WS_FRAME_INTERVAL_MS).toBe(200);
  });

  it('allows max 10 reconnect attempts', () => {
    expect(MAX_RECONNECT_ATTEMPTS).toBe(10);
  });

  it('defines a base reconnect delay', () => {
    expect(RECONNECT_BASE_DELAY_MS).toBe(500);
  });

  it('config module has all required exports', () => {
    const config = require('../config/backend');
    expect(config.BACKEND_URL).toBeDefined();
    expect(config.WS_FRAME_INTERVAL_MS).toBeDefined();
    expect(config.MAX_RECONNECT_ATTEMPTS).toBeDefined();
    expect(config.RECONNECT_BASE_DELAY_MS).toBeDefined();
  });
});
