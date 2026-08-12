import Foundation
import CoreVideo
import UIKit

//Frame Encoding and Compression Component
class FrameEncoderManager{
    
    //Compression Quality Setting (0.0 to 1.0)
    private let compressionQuality:CGFloat
    
    //Maximum Image Side Length for Model Preprocessing
    private let targetWidth:CGFloat
    
    init(compressionQuality: CGFloat=0.5, targetWidth: CGFloat=640.0) {
        self.compressionQuality = compressionQuality
        self.targetWidth = targetWidth
    }
    
    //Core Processing Model
    
    // Convert Raw Pixel Buffer to Compressed Binary Payload
    func encode(pixelBuffer:CVPixelBuffer)->Data?{
        
        //Wrap memory pointer into a Core Image reference
        let ciImage=CIImage(cvPixelBuffer: pixelBuffer)
        
        //Calculation of scale factor to shrink dimensions
        let currentWidth=ciImage.extent.width
        let scale=targetWidth/currentWidth
        
        //Downscale image using CI filter(reduces tensor size for Python)
        let transformedImage=ciImage.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        
        //Render CoreImage into Standard UIImage
        let context=CIContext(options: [.useSoftwareRenderer:false])
        guard let cgImage=context.createCGImage(transformedImage, from: transformedImage.extent)
        else{
            return nil
        }
        let uiImage=UIImage(cgImage: cgImage)
        
        // Compress into JPEG Binary Data
        return uiImage.jpegData(compressionQuality:compressionQuality)
    }
}
