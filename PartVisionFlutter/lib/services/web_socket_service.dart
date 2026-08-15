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

  List<Detection> get detections => List.unmodifiable(_detections);
  double get processTimeMs => _processTimeMs;
  String get status => _status;
  String? get error => _error;

  bool get isConnected => _status == 'connected';

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

      _subscription = channel.stream.listen(
        (data) {
          debugPrint('[WS] Received: ${data is String ? "JSON" : "binary"}');
          try {
            if (data is String) {
              final json = jsonDecode(data) as Map<String, dynamic>;
              final detectionsList = json['detections'] as List<dynamic>? ?? [];
              _detections.clear();
              _detections.addAll(
                detectionsList.map((d) => Detection.fromJson(d as Map<String, dynamic>)),
              );
              _processTimeMs = (json['process_time_ms'] as num?)?.toDouble() ?? 0;
              _error = null;
              notifyListeners();
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
