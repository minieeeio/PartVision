import { useState, useEffect, useRef, useCallback } from 'react';
import { PartDetection, DetectionResponse } from '../types/detection';
import { EncodedFrameData } from '../utils/frameEncoder';
import {
  WS_FRAME_INTERVAL_MS,
  MAX_RECONNECT_ATTEMPTS,
  RECONNECT_BASE_DELAY_MS,
} from '../config/backend';

export const useWebSocket = (url: string) => {
  const [isConnected, setIsConnected] = useState(false);
  const [detections, setDetections] = useState<PartDetection[]>([]);
  const ws = useRef<WebSocket | null>(null);
  const reconnectAttempts = useRef(0);
  const reconnectTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);
  const connectFn = useRef<() => void>(() => {});
  const lastSendTime = useRef(0);
  const isSendingFrame = useRef(false);

  const sendFrame = useCallback((encodedData: EncodedFrameData) => {
    if (ws.current?.readyState === WebSocket.OPEN && !isSendingFrame.current) {
      const now = Date.now();
      if (now - lastSendTime.current < WS_FRAME_INTERVAL_MS) {
        return;
      }
      lastSendTime.current = now;
      isSendingFrame.current = true;
      console.log(`[WebSocket] Sending frame: ${encodedData.width}x${encodedData.height}, ${encodedData.buffer.byteLength} bytes`);
      ws.current.send(encodedData.buffer);
    } else if (ws.current?.readyState !== WebSocket.OPEN) {
      console.log(`[WebSocket] Cannot send frame - socket state: ${ws.current?.readyState}`);
    }
  }, []);

  const connect = useCallback(() => {
    console.log(`[WebSocket] Attempting connection to ${url}...`);
    const socket = new WebSocket(url);
    ws.current = socket;

    socket.onopen = () => {
      console.log('[WebSocket] Connection opened');
      setIsConnected(true);
      reconnectAttempts.current = 0;
    };

    socket.onmessage = (event) => {
      try {
        const data: DetectionResponse = JSON.parse(event.data);
        console.log(`[WebSocket] Received ${data.detections?.length || 0} detections`);
        setDetections(data.detections);
      } catch (err) {
        console.error('[WebSocket] JSON Parse Error:', err);
      } finally {
        isSendingFrame.current = false;
      }
    };

    socket.onerror = (error) => {
      console.error('[WebSocket] Socket error:', error);
    };

    socket.onclose = (event) => {
      console.log(`[WebSocket] Connection closed: code=${event.code}, reason=${event.reason}`);
      setIsConnected(false);
      isSendingFrame.current = false;

      if (reconnectAttempts.current < MAX_RECONNECT_ATTEMPTS) {
        const delay = RECONNECT_BASE_DELAY_MS * Math.pow(2, reconnectAttempts.current);
        reconnectAttempts.current += 1;
        console.log(`[WebSocket] Reconnecting in ${delay}ms (attempt ${reconnectAttempts.current})`);
        reconnectTimeout.current = setTimeout(() => {
          connectFn.current();
        }, delay);
      } else {
        console.log('[WebSocket] Max reconnect attempts reached');
      }
    };
  }, [url]);

  useEffect(() => {
    connectFn.current = connect;
  }, [connect]);

  useEffect(() => {
    console.log(`[WebSocket] Mount effect running, calling connectFn...`);
    connectFn.current();

    return () => {
      console.log(`[WebSocket] Cleanup effect running, closing socket...`);
      if (reconnectTimeout.current) {
        clearTimeout(reconnectTimeout.current);
        reconnectTimeout.current = null;
      }
      if (ws.current) {
        ws.current.close();
        ws.current = null;
      }
    };
  }, [connect]);

  return { isConnected, detections, sendFrame };
};
