import SwiftUI

// MARK: - Sci-Fi HUD Header & Action Overlay Component
struct ScannerHUDView: View {
    
    // 1. Connection status flag passed from parent
    let isConnected: Bool
    
    // 2. Action callback triggered when yellow SCANNER button is tapped
    var onScanTapped: () -> Void
    
    var body: some View {
        VStack {
            // MARK: - Top Industrial Header Bar
            HStack(spacing: 12) {
                Image(systemName: "wrench.and.screwdriver.fill")
                    .font(.system(size: 16, weight: .bold))
                
                Text("CORE_SCAN_V1.0")
                    .font(.system(size: 18, weight: .black, design: .monospaced))
                    .tracking(1.5)
                
                Spacer()
                
                // Network Status Indicator Dot
                Circle()
                    .fill(isConnected ? Color.green : Color.red)
                    .frame(width: 10, height: 10)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .background(Color.white)
            .foregroundColor(.black)
            .cornerRadius(4)
            .shadow(color: .black.opacity(0.3), radius: 4, x: 0, y: 2)
            .padding(.horizontal, 16)
            .padding(.top, 10)
            
            Spacer()
            
            // MARK: - Bottom Action Control Bar
            VStack {
                Button(action: {
                    onScanTapped()
                }) {
                    VStack(spacing: 4) {
                        Image(systemName: "camera.fill")
                            .font(.system(size: 18, weight: .bold))
                        
                        Text("SCANNER")
                            .font(.system(size: 11, weight: .black, design: .monospaced))
                    }
                    .foregroundColor(.black)
                    .frame(width: 110, height: 54)
                    .background(Color.yellow)
                    .border(Color.black, width: 2.5)
                    .shadow(color: .black.opacity(0.4), radius: 0, x: 3, y: 3)
                }
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 20)
            .background(Color.white.opacity(0.95))
        }
    }
}

