import { useState, useEffect, useRef, useCallback } from 'react';
import { PartDetection, DetectionResponse } from '../types/detection';

export const useWebSocket = (url: string) => {
  const [isConnected, setIsConnected] = useState(false);
  const [detections, setDetections] = useState<PartDetection[]>([]);
  const ws = useRef<WebSocket | null>(null);
  const isSendingFrame = useRef(false);

  useEffect(() => {
    const socket = new WebSocket(url);
    ws.current = socket;

    socket.onopen = () => {
      console.log('[WebSocket] Connected to backend');
      setIsConnected(true);
    };

    socket.onmessage = (event) => {
      try {
        const data: DetectionResponse = JSON.parse(event.data);
        setDetections(data.detections);
      } catch (err) {
        console.error('[WebSocket] JSON Parse Error:', err);
      } finally {
        isSendingFrame.current = false;
      }
    };

    socket.onerror = (error) => {
      console.error('[WebSocket] Error:', error);
    };

    socket.onclose = () => {
      console.log('[WebSocket] Connection closed');
      setIsConnected(false);
    };

    return () => {
      socket.close();
    };
  }, [url]);

  const sendFrame = useCallback((base64Data: string) => {
    if (ws.current?.readyState === WebSocket.OPEN && !isSendingFrame.current) {
      isSendingFrame.current = true;
      ws.current.send(base64Data);
    }
  }, []);

  return { isConnected, detections, sendFrame };
};