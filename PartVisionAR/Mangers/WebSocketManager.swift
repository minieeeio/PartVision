import Foundation
import Combine


class WebSocketManager:ObservableObject{
    @Published var latestDetections:[PartDetection]=[]
    @Published var isConnected:Bool=false
    
    private var webSocketTask:URLSessionWebSocketTask?
    
    private var isSendingFrame:Bool=false
    
    func connect(urlString:String){
        guard let url=URL(string:urlString)
        else{
            return
        }
        
        let session=URLSession(configuration: .default)
        webSocketTask=session.webSocketTask(with: url)
        webSocketTask?.resume()
        
        DispatchQueue.main.async{
            
        }
    }
    
    func disconnect(){
        webSocketTask?.cancel(with: .goingAway, reason: nil)
        DispatchQueue.main.async{
            self.isConnected=false
        }
    }
    
    func sendFrameData(_ frameData:Data){
        
        guard isConnected,!isSendingFrame,let task=webSocketTask else{
            return
        }
        
        isSendingFrame=true
        
        let message=URLSessionWebSocketTask.Message.data(frameData)
        
        task.send(message){[weak self] error in
            DispatchQueue.main.async{
                self?.isSendingFrame=false
            }
            if let error=error{
                print("WebSocket send error: \(error.localizedDescription)")
            }
            
        }
        
    }
    
    private func receiveData() {
            webSocketTask?.receive { [weak self] result in
                guard let self = self else { return }
                
                switch result {
                case .success(let message):
                    switch message {
                    case .string(let jsonString):
                        self.parseDetectionJSON(jsonString)
                    case .data(let data):
                        if let jsonString = String(data: data, encoding: .utf8) {
                            self.parseDetectionJSON(jsonString)
                        }
                    @unknown default:
                        break
                    }
                    
                    // Keep the listening loop active for the next incoming response
                    self.receiveData()
                    
                case .failure(let error):
                    print("WebSocket receive error: \(error.localizedDescription)")
                    DispatchQueue.main.async {
                        self.isConnected = false
                    }
                }
            }
        }
    
    
    private func parseDetectionJSON(_ jsonString: String) {
            guard let data = jsonString.data(using: .utf8) else { return }
            
            do {
                let decoder = JSONDecoder()
                let response = try decoder.decode(DetectionResponse.self, from: data)
                
                // UI updates must always run on the Main Thread
                DispatchQueue.main.async {
                    self.latestDetections = response.detections
                }
            } catch {
                print("JSON Parsing error: \(error)")
            }
        }
}

