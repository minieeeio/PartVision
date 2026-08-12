import {
  ConnectionState,
  DetectionResponse,
  PartDetection,
} from '../models/DetectionModel';
import { WEBSOCKET_CONFIG, BACKEND_URL } from '../utils/constants';

export type StateListener = (state: ConnectionState) => void;
export type DetectionListener = (detections: PartDetection[]) => void;
export type ErrorListener = (message: string) => void;

export class WebSocketManager {
  private ws: WebSocket | null = null;
  private url: string = BACKEND_URL;
  private reconnectAttempts = 0;
  private reconnectTimeout: ReturnType<typeof setTimeout> | null = null;

  private state: ConnectionState = 'disconnected';
  private detections: PartDetection[] = [];
  private sending = false;

  private stateListeners: StateListener[] = [];
  private detectionListeners: DetectionListener[] = [];
  private errorListeners: ErrorListener[] = [];

  connect(): void {
    if (this.state === 'connecting' || this.state === 'connected') return;

    this.state = 'connecting';
    this.notifyState();

    try {
      this.ws = new WebSocket(this.url);
      this.ws.onopen = this.onOpen.bind(this);
      this.ws.onmessage = this.onMessage.bind(this);
      this.ws.onerror = this.onError.bind(this);
      this.ws.onclose = this.onClose.bind(this);
    } catch (e) {
      this.notifyError(`WebSocket init failed: ${e}`);
    }
  }

  private onOpen(): void {
    this.state = 'connected';
    this.reconnectAttempts = 0;
    this.notifyState();
  }

  private onMessage(event: { data: string | ArrayBuffer }): void {
    const json =
      typeof event.data === 'string'
        ? event.data
        : new TextDecoder().decode(new Uint8Array(event.data));

    try {
      const resp: DetectionResponse = JSON.parse(json);
      this.detections = resp.detections;
      this.notifyDetections(resp.detections);
    } catch (e) {
      this.notifyError(`JSON parse: ${e}`);
    }
  }

  private onError(event: any): void {
    const msg = event?.message ?? 'WebSocket error';
    this.notifyError(msg);
  }

  private onClose(): void {
    this.ws = null;
    const wasConnected = this.state === 'connected';
    this.state = 'disconnected';
    this.notifyState();
    if (wasConnected) this.scheduleReconnect();
  }

  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= WEBSOCKET_CONFIG.MAX_RECONNECT_ATTEMPTS) {
      this.notifyError('Max reconnect attempts reached');
      return;
    }
    const delay =
      WEBSOCKET_CONFIG.RECONNECT_INTERVAL_MS *
      Math.pow(1.5, this.reconnectAttempts);
    this.reconnectAttempts++;
    this.reconnectTimeout = setTimeout(() => this.connect(), delay);
  }

  sendFrame(data: Uint8Array | ArrayBuffer): void {
    if (!this.ws || this.sending || this.state !== 'connected') return;
    this.sending = true;
    try {
      this.ws.send(data);
    } catch (e) {
      this.notifyError(`Send failed: ${e}`);
    } finally {
      this.sending = false;
    }
  }

  disconnect(): void {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
    this.ws?.close();
    this.ws = null;
    this.state = 'disconnected';
    this.notifyState();
  }

  getDetections(): PartDetection[] {
    return this.detections;
  }

  getState(): ConnectionState {
    return this.state;
  }

  onState = (cb: StateListener): void => {
    this.stateListeners.push(cb);
  };
  onDetections = (cb: DetectionListener): void => {
    this.detectionListeners.push(cb);
  };
  onError = (cb: ErrorListener): void => {
    this.errorListeners.push(cb);
  };

  private notifyState(): void {
    this.stateListeners.forEach((l) => l(this.state));
  }
  private notifyDetections(d: PartDetection[]): void {
    this.detectionListeners.forEach((l) => l(d));
  }
  private notifyError(msg: string): void {
    this.errorListeners.forEach((l) => l(msg));
  }
}
