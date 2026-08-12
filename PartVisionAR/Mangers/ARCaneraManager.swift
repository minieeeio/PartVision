import Foundation
import ARKit
import Combine


//AR Camera Contoller Manager
class ARCameraManager: NSObject,ObservableObject,ARSessionDelegate{
    
    //Hardware Render View
    let arView:ARSCNView=ARSCNView(frame: .zero)
    
    //Continious Data Hand-off Event (Closure)
    var onFrameCaptured:((CVPixelBuffer)->Void)?
    
    override init(){
        super.init()
        //connect this manager to listen directly to hardware camera events
        arView.session.delegate=self
        startSession()
    }
    
    
    //session management cycle
    func startSession(){
        //sensor tracking strategy configuration
        let configuration=ARWorldTrackingConfiguration()
        configuration.isAutoFocusEnabled=true
        
        //run hardware engine
        arView.session.run(configuration,options:[.resetTracking,.removeExistingAnchors])
        
    }
    
    func pauseSession(){
        //Pause hardware tracking to save battery when screen is hidden
        arView.session.pause()
    }
    
    
    //High-Frequency Frame Extractor Callback
    func session(_ session:ARSession,didUpdate frame:ARFrame){
        //Extract raw image buffer stored in unified GPU Memory
        let  pixelBuffer=frame.capturedImage
        
        // Forward buffer directly to our compression/network pipeline
        onFrameCaptured?(pixelBuffer)
    }
    
}



