import SwiftUI

// MARK: - Main Application Scanner Screen
struct ScannerContainerView: View {
    
    // 1. Core Lifecycle & Manager Instances
    @StateObject private var cameraManager = ARCameraManager()
    @StateObject private var webSocketManager = WebSocketManager()
    
    // 2. Data Compressor Instance (50% JPEG quality, downscaled to 640px)
    private let encoderManager = FrameEncoderManager(compressionQuality: 0.5, targetWidth: 640.0)
    
    // Backend WebSocket Endpoint URL
    private let backendURLString = "wss://your-fastapi-server.com/ws/segment"
    
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Layer A: Live AR Camera Background
                ARViewContainer(cameraManager: cameraManager)
                    .edgesIgnoringSafeArea(.all)
                
                // Layer B: Real-Time Bounding Box Overlays
                ForEach(webSocketManager.latestDetections) { detection in
                    BoundingBoxView(
                        detection: detection,
                        containerSize: geometry.size
                    )
                }
                
                // Layer C: Top Header HUD & Bottom Action Button
                ScannerHUDView(
                    isConnected: webSocketManager.isConnected,
                    onScanTapped: {
                        // Action performed when user taps the yellow SCANNER button
                        print("Manual scan triggered")
                    }
                )
            }
        }
        // MARK: - Lifecycle Hooks & Streaming Pipeline Setup
        .onAppear {
            setupStreamingPipeline()
            webSocketManager.connect(urlString: backendURLString)
        }
        .onDisappear {
            webSocketManager.disconnect()
            cameraManager.pauseSession()
        }
    }
    
    // MARK: - Pipeline Wiring
    
    // Connects ARKit camera frames directly to the encoder and socket transport
    private func setupStreamingPipeline() {
        cameraManager.onFrameCaptured = { [weak webSocketManager] pixelBuffer in
            // Step 1: Compress raw pixel buffer into lightweight binary JPEG data
            guard let compressedData = self.encoderManager.encode(pixelBuffer: pixelBuffer) else { return }
            
            // Step 2: Transmit bytes over WebSocket pipe
            webSocketManager?.sendFrameData(compressedData)
        }
    }
}

#Preview {
    ScannerContainerView()
}

