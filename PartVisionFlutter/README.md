# PartVision Flutter

Real-time car-part segmentation and detection. A Flutter mobile client sends camera frames over WebSocket to a FastAPI + PyTorch backend, which returns normalized bounding boxes and segmentation polygons for car-part classes.

## Architecture

```
┌─────────────┐        WebSocket         ┌──────────────────┐
│  Flutter     │ ───────────────────────► │  FastAPI Backend │
│  (Dart)      │  binary JPEG + JSON     │  (PyTorch/ONNX)  │
└─────────────┘                         └──────────────────┘
```

## Prerequisites

- Flutter SDK 3.x+
- Android Studio / Xcode
- Backend server running at `http://YOUR_PC_IP:5555`

## Setup

### 1. Install dependencies

```bash
cd PartVisionFlutter
flutter pub get
```

### 2. Configure backend URL

The app resolves the backend URL from remote config by default.
For local testing, update the remote config file or modify `lib/services/config_service.dart` to hardcode your backend IP.

### 3. Run on device

```bash
flutter run
```

### 4. Build APK

```bash
flutter build apk --release
```

## Design System

Inspired by Apple's photography-first interface:

- **Primary**: Action Blue `#0066cc`
- **Surface**: Pure black `#000000` for immersive camera view
- **Typography**: SF Pro Display / SF Pro Text with tight tracking on headlines
- **HUD**: Minimalist status indicators with translucent backgrounds
- **Overlay**: Bright cyan bounding boxes with polygon masks

## WebSocket Protocol

**Client → Server**: Binary JPEG bytes

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
      "polygon": [{"x": 0.12, "y": 0.34}, ...]
    }
  ],
  "process_time_ms": 45.2
}
```

All coordinates are normalized `0.0` – `1.0`.
