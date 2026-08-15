import 'dart:async';
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import '../models/detection.dart';
import 'config_service.dart';

class WebSocketService extends ChangeNotifier {
  WebSocketChannel? _channel;
  StreamSubscription? _subscription;
  final List<Detection> _detections = [];
  double _processTimeMs = 0;
  String _status = 'disconnected';
  String? _error;
  String _selectedModel = 'partlitunet';
  VoidCallback? _onConnected;

  List<Detection> get detections => List.unmodifiable(_detections);
  double get processTimeMs => _processTimeMs;
  String get status => _status;
  String? get error => _error;
  String get selectedModel => _selectedModel;

  bool get isConnected => _status == 'connected';

  void setSelectedModel(String model) {
    _selectedModel = model;
    if (isConnected) {
      sendControlMessage({'type': 'switch_model', 'model_type': model});
    }
  }

  void onConnected(VoidCallback callback) {
    _onConnected = callback;
  }

  Future<void> connect(String apiBaseUrl) async {
    disconnect();
    try {
      final wsUrl = ConfigService.resolveWebSocketUrl(apiBaseUrl);
      debugPrint('[WS] Connecting to: $wsUrl');

      _status = 'connecting';
      _error = null;
      notifyListeners();

      final channel = WebSocketChannel.connect(Uri.parse(wsUrl));

      await channel.ready.timeout(
        const Duration(seconds: 5),
        onTimeout: () {
          debugPrint('[WS] Connection timeout after 5s');
          throw TimeoutException('WebSocket connection timeout');
        },
      );

      _channel = channel;
      _status = 'connected';
      _error = null;
      notifyListeners();

      if (_selectedModel.isNotEmpty) {
        sendControlMessage({'type': 'switch_model', 'model_type': _selectedModel});
      }
      _onConnected?.call();

      _subscription = channel.stream.listen(
        (data) {
          debugPrint('[WS] Received: ${data is String ? "JSON" : "binary"}');
          try {
            if (data is String) {
              final json = jsonDecode(data) as Map<String, dynamic>;
              final msgType = json['type'];
              if (msgType == 'model_switched') {
                debugPrint('[WS] Model switched: ${json['current_model']}');
                final switchResult = json['switch_result'];
                if (switchResult != null && switchResult['status'] == 'error') {
                  debugPrint('[WS] Model switch error: ${switchResult['message']}');
                  final detail = switchResult['detail'];
                  if (detail != null) {
                    debugPrint('[WS] Model switch detail: $detail');
                  }
                  _error = '${switchResult['message']}${detail != null ? ": $detail" : ""}';
                  notifyListeners();
                }
              } else {
                final detectionsList = json['detections'] as List<dynamic>? ?? [];
                _detections.clear();
                _detections.addAll(
                  detectionsList.map((d) => Detection.fromJson(d as Map<String, dynamic>)),
                );
                _processTimeMs = (json['process_time_ms'] as num?)?.toDouble() ?? 0;
                _error = null;
                notifyListeners();
              }
            }
          } catch (e) {
            _error = 'Parse error: $e';
            notifyListeners();
          }
        },
        onError: (error) {
          debugPrint('[WS] Stream error: $error');
          _status = 'error';
          _error = error.toString();
          notifyListeners();
        },
        onDone: () {
          debugPrint('[WS] Stream closed');
          _status = 'disconnected';
          notifyListeners();
        },
        cancelOnError: false,
      );

      debugPrint('[WS] Connected successfully');
    } on TimeoutException catch (e) {
      debugPrint('[WS] Timeout: $e');
      _status = 'error';
      _error = 'Connection timeout';
      notifyListeners();
    } catch (e) {
      debugPrint('[WS] Connection failed: $e');
      _status = 'error';
      _error = e.toString();
      notifyListeners();
    }
  }

  Future<void> sendFrame(Uint8List jpegBytes) async {
    if (_channel == null || _status != 'connected') return;
    try {
      debugPrint('[WS] Sending frame: ${jpegBytes.length} bytes');
      _channel!.sink.add(jpegBytes);
    } catch (e) {
      debugPrint('[WS] Send error: $e');
      _error = 'Send error: $e';
      notifyListeners();
    }
  }

  Future<void> sendControlMessage(Map<String, dynamic> message) async {
    if (_channel == null || _status != 'connected') return;
    try {
      final jsonStr = jsonEncode(message);
      debugPrint('[WS] Sending control: $jsonStr');
      _channel!.sink.add(jsonStr);
    } catch (e) {
      debugPrint('[WS] Control send error: $e');
      _error = 'Send error: $e';
      notifyListeners();
    }
  }

  void disconnect() {
    try {
      _subscription?.cancel();
    } catch (_) {}
    try {
      _channel?.sink.close(1000, 'Client disconnecting');
    } catch (_) {}
    _subscription = null;
    _channel = null;
    _status = 'disconnected';
    notifyListeners();
  }

  @override
  void dispose() {
    disconnect();
    super.dispose();
  }
}
