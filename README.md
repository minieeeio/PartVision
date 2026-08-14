# PartVision

Real-time car-part segmentation and detection system. A React Native mobile client sends camera frames over WebSocket to a FastAPI + PyTorch backend, which returns normalized bounding boxes and segmentation polygons for 23 car-part classes.

## What it does

- Captures live camera frames on an Android device
- Streams JPEG frames to a Python backend via WebSocket
- Runs PartLiteUNet segmentation (24 classes including background)
- Returns detections with normalized coordinates, confidence scores, and polygon masks
- Overlays detection results on the camera preview in real time

## Architecture

```
┌─────────────┐        WebSocket         ┌──────────────────┐
│  React Native │ ───────────────────────► │  FastAPI Backend │
│  (Expo/TS)   │  binary JPEG + JSON     │  (PyTorch/ONNX)  │
└─────────────┘                         └──────────────────┘
```

**Frontend** (`PartVision/`): Expo SDK ~57, React 19, React Native 0.86, `react-native-vision-camera`, `expo-router`.

**Backend** (`PartVisionBackend/`): FastAPI, uvicorn, OpenCV, ONNX Runtime or PyTorch, rembg for background removal.

## Prerequisites

- **Frontend**: Node.js 18+, JDK 17, Android Studio + SDK (for APK build)
- **Backend**: Python 3.11, NVIDIA GPU with CUDA 12.8 (cu128)
- **Model weights**: Place `best.onnx` (or `best.pt`/`best.pth`) in `PartVisionBackend/weights/`

---

## Backend Setup

### 1. Create Python 3.11 environment with CUDA 12.8

```bash
cd PartVisionBackend

# Create a fresh Python 3.11 environment
conda create -n partvision python=3.11 -y
conda activate partvision

# Install PyTorch with CUDA 12.8 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Verify CUDA is visible
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### 2. Install backend dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` pins:
- fastapi, uvicorn, websockets
- opencv-python-headless, numpy, pillow
- pydantic, pydantic-settings
- onnxruntime
- rembg

### 3. Add model weights

Place your trained model in `PartVisionBackend/weights/`:
- `best.onnx` (preferred) or `best.pt` / `best.pth`

The backend auto-detects the model path on startup via `config.py`.

### 4. Run the server

```bash
python main.py
```

The server starts on `0.0.0.0:5555` and exposes:
- `GET /health` — server status, model info, system resources
- `GET /metrics` — inference latency, throughput, CPU/memory
- `WS /ws/segment` — persistent WebSocket for frame streaming and detection results

### 5. Configure the frontend to reach the backend

The mobile app resolves the backend URL from:
1. `app.json` `expo.extra.backendUrl`
2. Remote config at `https://raw.githubusercontent.com/PrageshShrestha/hasslefree/main/partVision`

For local testing, update `PartVision/app.json`:
```json
{
  "expo": {
    "extra": {
      "backendUrl": "ws://YOUR_PC_IP:5555/ws/segment"
    }
  }
}
```

For production, update the remote config file to point to your deployed backend (e.g. Zrok tunnel, ngrok, or public IP).

---

## Frontend Setup (React Native / Expo)

### 1. Install dependencies

```bash
cd PartVision
npm install
```

### 2. Start the development server

```bash
npm start
```

Expo DevTools opens. Press `a` to run on an Android emulator or `w` for web.

### 3. Configure camera and permissions

The app requests `CAMERA` permission on first launch. Ensure `app.json` includes:
```json
{
  "expo": {
    "android": {
      "permissions": ["android.permission.CAMERA"]
    }
  }
}
```

---

## Build Android APK

### Option A: EAS Build (recommended)

1. Install EAS CLI:
   ```bash
   npm install -g eas-cli
   ```

2. Configure EAS:
   ```bash
   eas build:configure
   ```

3. Build APK:
   ```bash
   eas build --platform android --profile preview
   ```

4. Download the APK from the EAS dashboard and install on your device.

### Option B: Local build with Gradle

1. Pre-build the native Android project:
   ```bash
   npx expo prebuild --platform android
   ```

2. Build the APK:
   ```bash
   cd android
   ./gradlew assembleRelease
   ```

3. The APK is at `android/app/build/outputs/apk/release/app-release.apk`.

4. Install to a connected device:
   ```bash
   adb install android/app/build/outputs/apk/release/app-release.apk
   ```

### Option C: Development build (faster iteration)

```bash
npx expo run:android
```

This compiles and installs a debug APK directly to a connected device or emulator. Faster than EAS for development, but the binary is larger and not optimized for distribution.

---

## WebSocket Protocol

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

---

## Project Structure

```
PartVision/
├── PartVision/                  # Expo React Native app
│   ├── app/                     # Screens and routes (expo-router)
│   ├── src/
│   │   ├── components/          # ARCameraView, PolygonOverlay, HUD
│   │   ├── config/              # Backend URL resolution
│   │   ├── hooks/               # WebSocket frame streaming
│   │   ├── screens/             # ScannerScreen
│   │   └── types/               # PartDetection, DetectionResponse
│   ├── assets/                  # Icons, splash
│   ├── android/                 # Pre-built native Android project
│   ├── App.tsx                  # Entry point
│   ├── app.json                 # Expo config
│   └── package.json
│
├── PartVisionBackend/           # FastAPI inference server
│   ├── api/
│   │   └── websocket.py         # WebSocket endpoint
│   ├── core/
│   │   ├── decoder.py           # JPEG frame decoder
│   │   ├── postprocess.py       # Mask processing → bbox/polygon
│   │   ├── metrics.py           # Latency, throughput, resource monitor
│   │   └── config.py            # Settings (INPUT_SIZE, thresholds)
│   ├── models/
│   │   ├── model_loader.py      # ONNX / PyTorch wrapper + letterbox
│   │   └── part_lite_unet.py    # Model architecture
│   ├── weights/                 # Place best.onnx here
│   ├── tests/                   # pytest suite
│   ├── main.py                  # Uvicorn entry point
│   └── requirements.txt
```

---

## Troubleshooting

**Backend won't start / model not loading**
- Ensure `weights/best.onnx` exists in `PartVisionBackend/weights/`
- Check CUDA visibility: `python -c "import torch; print(torch.cuda.is_available())"`
- For CPU-only mode, set `USE_CUDA=false` environment variable

**App can't connect to backend**
- Confirm the PC and Android device are on the same network
- Use your PC's LAN IP (not `localhost` or `127.0.0.1`) in `app.json`
- For internet access, expose the backend via Zrok / ngrok and update the remote config URL

**APK install fails (app not installed)**
- Uninstall any previous build first: `adb uninstall com.anonymous.PartVision`
- Ensure `adb devices` shows your device as `device`

**Camera permission denied**
- On Android 13+, ensure `android.permission.CAMERA` is in `app.json`
- Grant permission manually in device settings if the prompt was dismissed

---

## Tests

Backend tests use pytest + FastAPI TestClient.

```bash
cd PartVisionBackend
python3 -m pytest tests/ -v
```

Frontend tests:
```bash
cd PartVision
npm test
```
