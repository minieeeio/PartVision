import SwiftUI
import ARKit


// UIKit to SwiftUI Bridge Component
struct ARViewContainer:UIViewRepresentable{
   
    //refernce to our Camera Engine State
    @ObservedObject var cameraManager:ARCameraManager
    
    //UIKit View Lifecycle Initalization
    func makeUIView(context: Context) -> some UIView {
        // Return the existing ARSCNView instance created inside ARCameraManger
        return cameraManager.arView
    }
    
    //Dynamic SwiftUI View State Updates
    func updateUIView(_ uiView: UIViewType, context: Context) {
        //Intentionaly empty as ARSCNView updates its own frames internally
    }
}

