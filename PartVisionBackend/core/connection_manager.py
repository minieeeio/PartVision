import logging
import asyncio
from typing import Optional
from fastapi import WebSocket

logger = logging.getLogger("PartVision")


class ConnectionManager:
    """Tracks active WebSocket connections and provides lifecycle management."""

    def __init__(self):
        self._active_connections: dict[str, WebSocket] = {}
        self._connection_times: dict[str, float] = {}

    async def connect(self, websocket: WebSocket) -> str:
        await websocket.accept()
        client_id = str(id(websocket))
        self._active_connections[client_id] = websocket
        import time
        self._connection_times[client_id] = time.time()
        logger.info(
            "Client connected — total: %d, id=%s",
            len(self._active_connections),
            client_id,
        )
        return client_id

    def disconnect(self, websocket: WebSocket) -> None:
        client_id = str(id(websocket))
        self._active_connections.pop(client_id, None)
        self._connection_times.pop(client_id, None)
        logger.info(
            "Client disconnected — remaining: %d, id=%s",
            len(self._active_connections),
            client_id,
        )

    async def send_json(self, websocket: WebSocket, data: dict) -> bool:
        client_id = str(id(websocket))
        if client_id not in self._active_connections:
            return False
        try:
            await websocket.send_json(data)
            return True
        except Exception as e:
            logger.warning("Send failed for %s: %s", client_id, e)
            return False

    async def receive_bytes_with_timeout(
        self,
        websocket: WebSocket,
        timeout: float = 5.0,
    ) -> Optional[bytes]:
        try:
            data = await asyncio.wait_for(
                websocket.receive_bytes(),
                timeout=timeout,
            )
            return data
        except asyncio.TimeoutError:
            return None
        except Exception:
            return None

    @property
    def active_count(self) -> int:
        return len(self._active_connections)


connection_manager = ConnectionManager()
