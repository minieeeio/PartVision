# PartVision

Real-time car-part segmentation and detection system. A mobile client streams live camera frames to a FastAPI + PyTorch backend over WebSocket, which returns normalized bounding boxes and segmentation polygons for 23 car-part classes.

## What's in this repository

- `PartVisionBackend/` — FastAPI inference server (PyTorch/ONNX, OpenCV, WebSocket)
- `PartVisionFlutter/` — Flutter mobile client (camera, web_socket_channel, image)
- `PartVision/` — React Native / Expo mobile client (Expo, react-native-vision-camera)

## Status

Active development. Not for general release.

## Prerequisites

- **Frontend**: Node.js 18+, JDK 17, Android Studio + SDK
- **Backend**: Python 3.11, NVIDIA GPU with CUDA 12.8 (or CPU mode)
- **Model weights**: Place `best.onnx` (PartLiteUNet) or `best.pt` (YOLOv8-seg) in `PartVisionBackend/weights/`
- **YOLO**: `pip install ultralytics`

## Backend setup

```bash
cd PartVisionBackend

# Create Python 3.11 environment
conda create -n partvision python=3.11 -y
conda activate partvision

# Install PyTorch with CUDA 12.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install backend dependencies
pip install -r requirements.txt

# Install YOLO dependencies
pip install ultralytics
```

## Running the main server (PartLiteUNet)

Server starts on `http://0.0.0.0:5555` with WebSocket at `ws://0.0.0.0:5555/ws/segment`.

```bash
python main.py
```

For CPU-only mode:
```bash
USE_CUDA=false python main.py
```

## Running the YOLO server (YOLOv8-seg)

The YOLO server runs on a separate port (`5556`) and uses the same WebSocket protocol.

```bash
python yolo_main.py
```

For CPU-only mode:
```bash
USE_CUDA=false python yolo_main.py
```

Server starts on `http://0.0.0.0:5556` with WebSocket at `ws://0.0.0.0:5556/ws/segment`.

**YOLO model setup:**
- Place your YOLOv8-seg `best.pt` in `PartVisionBackend/weights/`
- The server auto-detects `.pt` files and uses the YOLO wrapper
- Supported models: YOLOv8-seg, YOLOv11-seg (any segmentation model trained with Ultralytics)

**YOLO health endpoint:**
```bash
curl http://localhost:5556/health
```

## Frontend setup (Flutter)

```bash
cd PartVisionFlutter
flutter pub get
flutter run
```

## Frontend setup (React Native / Expo)

```bash
cd PartVision
npm install
npm start
```

## WebSocket protocol

**Client → Server**: Binary JPEG bytes (one frame per message)

**Server → Client**: JSON payload
```json
{
  "detections": [
    {
      "label": "FRONT_BUMPER",
      "confidence": 0.92,
      "x_min": 0.12,
      "y_min": 0.34,
      "width": 0.45,
      "height": 0.22,
      "polygon": [
        {"x": 0.12, "y": 0.34},
        {"x": 0.57, "y": 0.34},
        {"x": 0.55, "y": 0.56},
        {"x": 0.14, "y": 0.56}
      ]
    }
  ],
  "process_time_ms": 45.2
}
```

All coordinates are normalized `0.0` – `1.0` relative to the original frame dimensions.

## Backend endpoints

### Main server (PartLiteUNet) — port 5555

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Server status, model info, system resources |
| GET | `/metrics` | Inference latency, throughput, CPU/memory |
| POST | `/location` | Receive GPS location updates |
| WS | `/ws/segment` | Persistent WebSocket for frame streaming and detection results |

### YOLO server (YOLOv8-seg) — port 5556

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Server status, model info, device |
| WS | `/ws/segment` | Persistent WebSocket for frame streaming and detection results |

## Mobile app configuration

The mobile client resolves the backend URL at runtime from remote config:
- Remote config: `https://raw.githubusercontent.com/PrageshShrestha/hasslefree/main/partVision.json`
- Field: `api_base_url`
- Fallback: `app.json` `expo.extra.backendUrl` (Expo) or manifest metadata (Flutter)

For local testing, update the remote config file or hardcode your backend IP in the frontend config service.

**Using with YOLO server:**
- Point `api_base_url` to the YOLO server address (default port 5556)
- Example: `http://192.168.1.100:5556`

## Building the Android APK

### Flutter (recommended)

```bash
cd PartVisionFlutter
flutter build apk --debug
# APK at: build/app/outputs/flutter-apk/app-debug.apk
```

### React Native / Expo

```bash
cd PartVision
npx expo prebuild --platform android
cd android && ./gradlew assembleDebug
# APK at: android/app/build/outputs/apk/debug/app-debug.apk
```

Or use EAS Build:
```bash
eas build --platform android --profile preview
```

## Project structure

```
PartVision/
├── PartVisionBackend/           # FastAPI inference server
│   ├── api/
│   │   ├── websocket.py         # WebSocket endpoint (PartLiteUNet)
│   │   ├── yolo_websocket.py    # WebSocket endpoint (YOLOv8-seg)
│   │   └── reconstruction.py    # Reconstruction session API
│   ├── core/
│   │   ├── decoder.py           # JPEG frame decoder
│   │   ├── postprocess.py       # Mask processing → bbox/polygon
│   │   ├── metrics.py           # Latency, throughput, resource monitor
│   │   └── reconstructor.py     # 3D reconstruction pipeline
│   ├── models/
│   │   ├── model_loader.py      # ONNX / PyTorch wrapper (PartLiteUNet)
│   │   ├── yolo_model_loader.py # YOLOv8-seg wrapper
│   │   └── part_lite_unet.py    # Model architecture
│   ├── weights/                 # Place best.onnx or best.pt here
│   ├── tests/                   # pytest suite
│   ├── main.py                  # Uvicorn entry point (PartLiteUNet)
│   ├── yolo_main.py             # Uvicorn entry point (YOLOv8-seg)
│   ├── yolo_config.py           # YOLO-specific settings
│   └── requirements.txt
├── PartVisionFlutter/           # Flutter mobile client
│   ├── lib/
│   │   ├── main.dart
│   │   ├── models/detection.dart
│   │   ├── services/
│   │   │   ├── camera_service.dart
│   │   │   ├── config_service.dart
│   │   │   └── web_socket_service.dart
│   │   ├── screens/camera_screen.dart
│   │   └── widgets/
│   │       ├── bounding_box_painter.dart
│   │       └── hud.dart
│   ├── android/
│   ├── ios/
│   └── pubspec.yaml
├── PartVision/                  # React Native / Expo mobile client
│   ├── App.tsx
│   ├── app.json
│   ├── package.json
│   └── src/
│       ├── types/index.ts
│       ├── config/api.ts
│       ├── hooks/useObjectDetection.ts
│       ├── screens/CameraScreen.tsx
│       └── components/BoundingBoxOverlay.tsx
└── README.md
```

## Points of contact

- Repository owner/maintainer: see git log for authors

## License

Internal use only. Not licensed for distribution.
