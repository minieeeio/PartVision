import Foundation
import CoreGraphics

//Top-Level Response container
struct DetectionResponse:Codable{
    //Array of car parts,processing-time of server
    let detections:[PartDetection]
    let processTimeMs:Double?
    
    //for mapping of python to swift
    enum CodingKeys:String,CodingKey{
        case detections
        case processTimeMs="process_time_ms"
    }
}

//Individual part bounding box part
struct PartDetection:Identifiable,Codable{
    let id:UUID=UUID() //for detecting every element uniquely
    let label:String
    let confidence:Double
    
    //normalized coordinate of the parts
    let xMin:Double
    let yMin:Double
    let width:Double
    let height:Double
    
    //mapping from swift to python
    //codingkeys for conversion of snake-case to camel-case
    enum CodingKeys:String,CodingKey{
        case label
        case confidence
        case xMin="x_min"
        case yMin="y_min"
        case width
        case height
    }
    
}
