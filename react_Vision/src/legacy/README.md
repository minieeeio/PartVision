# Legacy Managers

These files belong to the previous server-side inference architecture
(camera → JPEG → WebSocket → backend). They are **not wired** into the
current on-device flow (`InferenceManager` + ONNX Runtime) and are kept
here only as a reference in case a server-fallback mode is needed later.

- `WebSocketManager.ts` — WebSocket connection, heartbeat, reconnect
- `ConfigManager.ts` — Remote backend config via GitHub raw URL
- `FrameEncoder.ts` — JPEG encoding helpers
- `CameraManager.ts` — Camera/frame encoder orchestration

If you decide to remove them permanently, delete this folder and clean
up any lingering imports.
