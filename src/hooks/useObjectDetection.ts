import { useEffect, useRef, useState, useCallback } from 'react';
import { useCameraDevices, useFrameProcessor } from 'react-native-vision-camera';
import { runOnJS } from 'react-native-reanimated';
import { Detection, DetectionResponse } from '../types';
import { fetchRemoteConfig, resolveWebSocketUrl, getApiBaseUrl } from '../config/api';

const TARGET_FPS = 5;
const FRAME_INTERVAL_MS = 1000 / TARGET_FPS;

export function useObjectDetection() {
  const [detections, setDetections] = useState<Detection[]>([]);
  const [processTime, setProcessTime] = useState<number>(0);
  const [wsStatus, setWsStatus] = useState<string>('disconnected');
  const [apiBaseUrl, setApiBaseUrl] = useState<string>('');
  const [configError, setConfigError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const lastSendTime = useRef<number>(0);
  const isSending = useRef<boolean>(false);
  const reconnectTimeout = useRef<ReturnType<typeof setTimeout>>();
  const devices = useCameraDevices();
  const device = devices.back;

  const releaseSendLock = useCallback(() => {
    isSending.current = false;
  }, []);

  const sendFrame = useCallback(
    (base64Data: string) => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        runOnJS(releaseSendLock)();
        return;
      }

      try {
        wsRef.current.send(base64Data);
      } catch (error) {
        console.error('[WS] Failed to send frame:', error);
      } finally {
        runOnJS(releaseSendLock)();
      }
    },
    [releaseSendLock]
  );

  const connectWebSocket = useCallback(
    (url: string) => {
      if (wsRef.current) {
        wsRef.current.close();
      }

      const ws = new WebSocket(url);
      wsRef.current = ws;
      setWsStatus('connecting');

      ws.onopen = () => {
        setWsStatus('connected');
        console.log('[WS] Connected to', url);
      };

      ws.onmessage = (event) => {
        try {
          const data: DetectionResponse = JSON.parse(event.data);
          if (data.detections) {
            runOnJS(setDetections)(data.detections);
          }
          if (data.process_time_ms !== undefined) {
            runOnJS(setProcessTime)(data.process_time_ms);
          }
        } catch (error) {
          console.error('[WS] Failed to parse message:', error);
        }
      };

      ws.onerror = (error) => {
        console.error('[WS] Error:', error);
        setWsStatus('error');
      };

      ws.onclose = (event) => {
        setWsStatus('disconnected');
        console.log('[WS] Closed:', event.reason);
        if (!event.wasClean) {
          reconnectTimeout.current = setTimeout(() => {
            if (apiBaseUrl) {
              connectWebSocket(resolveWebSocketUrl(apiBaseUrl));
            }
          }, 3000);
        }
      };
    },
    [apiBaseUrl, connectWebSocket]
  );

  useEffect(() => {
    let cancelled = false;

    async function initConfig() {
      try {
        const remoteConfig = await fetchRemoteConfig();
        if (cancelled) return;
        const url = remoteConfig.api_base_url || getApiBaseUrl();
        if (!url) {
          setConfigError('No api_base_url found in remote config or app.json');
          return;
        }
        setApiBaseUrl(url);
        const wsUrl = resolveWebSocketUrl(url);
        connectWebSocket(wsUrl);
      } catch (error) {
        if (!cancelled) {
          setConfigError(error instanceof Error ? error.message : 'Config fetch failed');
        }
      }
    }

    initConfig();

    return () => {
      cancelled = true;
      if (reconnectTimeout.current) {
        clearTimeout(reconnectTimeout.current);
      }
      if (wsRef.current) {
        wsRef.current.close(1000, 'Component unmounted');
        wsRef.current = null;
      }
    };
  }, [connectWebSocket]);

  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet';
      const now = Date.now();
      if (now - lastSendTime.current < FRAME_INTERVAL_MS) {
        return;
      }
      if (isSending.current) {
        return;
      }
      lastSendTime.current = now;
      isSending.current = true;

      try {
        if (frame.toBase64) {
          const base64 = frame.toBase64('JPEG', 0.7);
          runOnJS(sendFrame)(base64);
        } else {
          console.warn('[FrameProcessor] frame.toBase64() unavailable. Install a frame encoder plugin.');
          runOnJS(releaseSendLock)();
        }
      } catch (error) {
        console.error('[FrameProcessor] Error:', error);
        runOnJS(releaseSendLock)();
      }
    },
    [sendFrame, releaseSendLock]
  );

  return {
    detections,
    processTime,
    wsStatus,
    apiBaseUrl,
    configError,
    frameProcessor,
    device,
  };
}
