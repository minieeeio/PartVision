import SwiftUI

// MARK: - AR Bounding Box & HUD Label Component
struct BoundingBoxView: View {
    
    // 1. Parsed detection data object passed from parent view
    let detection: PartDetection
    
    // 2. Full screen bounds provided by GeometryReader
    let containerSize: CGSize
    
    var body: some View {
        // Calculate exact pixel dimensions from normalized fractions (0.0 - 1.0)
        let rect = CGRect(
            x: detection.xMin * containerSize.width,
            y: detection.yMin * containerSize.height,
            width: detection.width * containerSize.width,
            height: detection.height * containerSize.height
        )
        
        // Dynamic styling matching design: Cyan for top labels, Yellow for larger components
        let boxColor: Color = detection.label.contains("BUMPER") || detection.label.contains("LIGHT") ? .cyan : .yellow
        
        ZStack(alignment: .topLeading) {
            // 3. Sci-Fi Bounding Box Outline
            Rectangle()
                .stroke(boxColor, lineWidth: 2)
                .frame(width: rect.width, height: rect.height)
            
            // 4. Header Badge (Label Title)
            VStack(alignment: .leading, spacing: 0) {
                Text(detection.label)
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .padding(.horizontal, 6)
                    .padding(.vertical, 3)
                    .background(boxColor)
                    .foregroundColor(.black)
                
                Spacer()
                
                // 5. Confidence Score Badge at Box Bottom
                HStack {
                    Spacer()
                    Text(String(format: "%.1f%%", detection.confidence * 100))
                        .font(.system(size: 9, weight: .bold, design: .monospaced))
                        .foregroundColor(boxColor)
                        .padding(2)
                        .background(Color.black.opacity(0.75))
                }
            }
            .frame(width: rect.width, height: rect.height)
        }
        // Position the box dynamically over the live video view
        .position(x: rect.midX, y: rect.midY)
    }
}

