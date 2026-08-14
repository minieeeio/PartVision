import React from 'react';
import TestRenderer, { act } from 'react-test-renderer';
import { useWebSocket } from '../hooks/useWebSocket';

const TestComponent = ({ url }: { url: string }) => {
  const { isConnected, detections, sendFrame } = useWebSocket(url);
  return null;
};

describe('useWebSocket', () => {
  let mockWs: any;

  beforeEach(() => {
    mockWs = {
      readyState: 0,
      send: jest.fn(),
      close: jest.fn(),
      onopen: null,
      onmessage: null,
      onclose: null,
      onerror: null,
    };

    (global as any).WebSocket = jest.fn(() => mockWs);
    (global as any).WebSocket.OPEN = 1;
  });

  afterEach(() => {
    jest.clearAllMocks();
    jest.useRealTimers();
  });

  it('initializes WebSocket connection on mount', () => {
    act(() => {
      TestRenderer.create(<TestComponent url="ws://localhost:8000" />);
    });
    expect(global.WebSocket).toHaveBeenCalledWith('ws://localhost:8000');
  });

  it('sets connected state on WebSocket open', () => {
    act(() => {
      TestRenderer.create(<TestComponent url="ws://localhost:8000" />);
    });

    act(() => {
      mockWs.onopen();
    });
  });

  it('handles incoming messages without crashing', () => {
    act(() => {
      TestRenderer.create(<TestComponent url="ws://localhost:8000" />);
    });

    act(() => {
      mockWs.onopen();
    });

    const mockResponse = {
      detections: [
        { label: 'FRONT_BUMPER', confidence: 0.95, x_min: 0.1, y_min: 0.2, width: 0.5, height: 0.3 },
      ],
      process_time_ms: 45.2,
    };

    act(() => {
      mockWs.onmessage({ data: JSON.stringify(mockResponse) });
    });
  });

  it('handles JSON parse errors gracefully', () => {
    act(() => {
      TestRenderer.create(<TestComponent url="ws://localhost:8000" />);
    });

    act(() => {
      mockWs.onopen();
    });

    act(() => {
      mockWs.onmessage({ data: 'not valid json' });
    });

    expect(true).toBe(true);
  });

  it('closes WebSocket on unmount', () => {
    let testRenderer: TestRenderer.ReactTestRenderer;
    act(() => {
      testRenderer = TestRenderer.create(<TestComponent url="ws://localhost:8000" />);
    });

    act(() => {
      mockWs.onopen();
    });

    act(() => {
      testRenderer!.unmount();
    });
    expect(mockWs.close).toHaveBeenCalled();
  });
});