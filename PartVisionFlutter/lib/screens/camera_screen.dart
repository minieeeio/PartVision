import 'dart:async';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';

import '../services/camera_service.dart';
import '../services/config_service.dart';
import '../services/web_socket_service.dart';
import '../widgets/bounding_box_painter.dart';
import '../widgets/hud.dart';

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> with WidgetsBindingObserver {
  CameraController? _controller;
  List<CameraDescription> _cameras = [];
  bool _isLoading = true;
  String? _configError;
  int _lastFrameTime = 0;
  final WebSocketService _wsService = WebSocketService();
  String? _apiBaseUrl;

  bool _isRecording = false;
  bool _showInfo = false;
  bool _isSendingFrame = false;
  String _selectedModel = 'partlitunet';
  bool _isCountdownActive = false;
  int _countdownValue = 3;
  Timer? _countdownTimer;
  bool _isConfiguring = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initialize();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _countdownTimer?.cancel();
    _stopRecording();
    _controller?.dispose();
    _wsService.dispose();
    super.dispose();
  }

  Future<void> _initialize() async {
    await _requestCameraPermission();
    await _loadCameras();
  }

  Future<void> _requestCameraPermission() async {
    final status = await Permission.camera.request();
    if (status.isPermanentlyDenied && mounted) {
      _showPermissionDenied();
    }
  }

  Future<void> _loadCameras() async {
    try {
      _cameras = await availableCameras();
      if (_cameras.isEmpty) {
        setState(() => _configError = 'No cameras found');
        return;
      }
      await _startCamera(_cameras.first);
    } catch (e) {
      setState(() => _configError = 'Camera error: $e');
    }
  }

  Future<void> _startCamera(CameraDescription description) async {
    _controller?.dispose();
    _controller = CameraController(
      description,
      ResolutionPreset.high,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    try {
      await _controller!.initialize();
      if (!mounted) return;
      setState(() {
        _isLoading = false;
        _configError = null;
      });
    } catch (e) {
      setState(() => _configError = 'Camera init error: $e');
    }
  }

  Future<void> _stopRecording() async {
    if (!_isRecording) return;

    try {
      await _controller?.stopImageStream();
    } catch (_) {}
    _wsService.disconnect();
    setState(() {
      _isRecording = false;
      _showInfo = false;
    });
  }

  Future<void> _fetchConfigAndConnect() async {
    try {
      debugPrint('[Camera] Fetching remote config...');
      final config = await ConfigService.fetchRemoteConfig();
      var baseUrl = config.apiBaseUrl;
      debugPrint('[Camera] Raw api_base_url from config: "$baseUrl"');

      if (baseUrl.isEmpty) {
        throw Exception('api_base_url is empty in remote config');
      }

      setState(() => _apiBaseUrl = baseUrl);

      final wsUrl = ConfigService.resolveWebSocketUrl(baseUrl);
      debugPrint('[Camera] Resolved WebSocket URL: $wsUrl');

      _wsService.connect(baseUrl);
    } catch (e) {
      debugPrint('[Camera] Config/connect error: $e');
      if (mounted) {
        setState(() => _configError = 'Config error: $e');
        _stopRecording();
      }
    }
  }

  void _onFrameAvailable(CameraImage image) async {
    if (!_isRecording) return;
    if (_isSendingFrame) return;

    final now = DateTime.now().millisecondsSinceEpoch;
    if (now - _lastFrameTime < FrameProcessor.frameIntervalMs) return;
    if (!_wsService.isConnected) {
      debugPrint('[Camera] Frame skipped: WebSocket not connected (status=${_wsService.status})');
      return;
    }

    _lastFrameTime = now;
    _isSendingFrame = true;

    try {
      final jpegBytes = await FrameProcessor.encodeJpeg(image);
      if (jpegBytes != null) {
        await _wsService.sendFrame(jpegBytes);
      }
    } catch (e) {
      debugPrint('[Camera] Frame processing error: $e');
    } finally {
      _isSendingFrame = false;
    }
  }

  void _toggleRecording() {
    if (_isRecording) {
      _stopRecording();
    } else {
      _startCountdownAndConnect();
    }
  }

  void _startCountdownAndConnect() {
    setState(() {
      _isCountdownActive = true;
      _countdownValue = 3;
      _isRecording = true;
    });

    _connectWebSocket();

    _countdownTimer = Timer.periodic(const Duration(seconds: 1), (timer) {
      if (_countdownValue > 1) {
        setState(() {
          _countdownValue--;
        });
      } else {
        timer.cancel();
        setState(() {
          _isCountdownActive = false;
        });
        _startImageStream();
      }
    });
  }

  Future<void> _connectWebSocket() async {
    if (_isConfiguring) {
      debugPrint('[Camera] Config already being fetched, waiting...');
      return;
    }

    try {
      if (_apiBaseUrl == null) {
        debugPrint('[Camera] No cached config, fetching before connect...');
        setState(() => _isConfiguring = true);
        await _fetchConfigAndConnect();
        setState(() => _isConfiguring = false);
      } else {
        final wsUrl = ConfigService.resolveWebSocketUrl(_apiBaseUrl!);
        debugPrint('[Camera] Connecting to WebSocket: $wsUrl');
        _wsService.onConnected(() {
          debugPrint('[Camera] WebSocket connected, sending initial model preference: $_selectedModel');
        });
        _wsService.setSelectedModel(_selectedModel);
        _wsService.connect(_apiBaseUrl!);
      }
    } catch (e) {
      setState(() => _isConfiguring = false);
      debugPrint('[Camera] Connection error: $e');
    }
  }

  void _startImageStream() {
    if (_controller == null || !_controller!.value.isInitialized) return;
    _controller!.startImageStream(_onFrameAvailable);
  }

  Future<void> _onModelChanged(String newModel) async {
    setState(() {
      _selectedModel = newModel;
    });

    debugPrint('[Camera] Model toggle changed to: $newModel');

    try {
      if (_apiBaseUrl == null) {
        debugPrint('[Camera] No cached config, fetching before model switch...');
        setState(() => _isConfiguring = true);
        final config = await ConfigService.fetchRemoteConfig();
        final baseUrl = config.apiBaseUrl;
        if (baseUrl.isEmpty) throw Exception('api_base_url is empty');
        setState(() {
          _apiBaseUrl = baseUrl;
          _isConfiguring = false;
        });
      }

      _wsService.setSelectedModel(newModel);
    } catch (e) {
      setState(() => _isConfiguring = false);
      debugPrint('[Camera] Model switch error: $e');
    }
  }

  void _showPermissionDenied() {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: const Color(0xFF1c1c1e),
        title: const Text('Camera Permission Required', style: TextStyle(color: Colors.white)),
        content: const Text('Please enable camera access in settings.', style: TextStyle(color: Color(0xFFcccccc))),
        actions: [
          TextButton(
            onPressed: () => openAppSettings(),
            child: const Text('Open Settings', style: TextStyle(color: Color(0xFF0066cc))),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return PopScope(
      canPop: false,
      child: Scaffold(
        backgroundColor: Colors.black,
        body: _buildBody(),
      ),
    );
  }

  Widget _buildBody() {
    if (_configError != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.error_outline, color: Color(0xFFFF4444), size: 48),
              const SizedBox(height: 16),
              Text(_configError!, textAlign: TextAlign.center, style: const TextStyle(color: Colors.white, fontSize: 16)),
              const SizedBox(height: 24),
              TextButton(
                onPressed: _initialize,
                child: const Text('Retry', style: TextStyle(color: Color(0xFF0066cc))),
              ),
            ],
          ),
        ),
      );
    }

    if (_isLoading || _controller == null || !_controller!.value.isInitialized) {
      return const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            CircularProgressIndicator(color: Color(0xFF0066cc)),
            SizedBox(height: 16),
            Text('Loading camera...', style: TextStyle(color: Colors.white)),
          ],
        ),
      );
    }

    final size = MediaQuery.of(context).size;

    Widget cameraPreview = SizedBox.expand(
      child: CameraPreview(_controller!),
    );

    return Stack(
      fit: StackFit.expand,
      children: [
        cameraPreview,
        ListenableBuilder(
          listenable: _wsService,
          builder: (context, _) {
            return CustomPaint(
              painter: BoundingBoxPainter(
                detections: _wsService.detections,
                frameSize: Size(_controller!.value.previewSize!.width, _controller!.value.previewSize!.height),
                viewSize: size,
              ),
              size: size,
            );
          },
        ),
        if (_isCountdownActive)
          Positioned.fill(
            child: _CountdownOverlay(count: _countdownValue),
          ),
        if (_isRecording && _wsService.status == 'connecting')
          const Positioned(
            top: 48,
            left: 16,
            child: _ConnectingIndicator(),
          ),
        if (_isRecording)
          Positioned(
            top: _wsService.status == 'connecting' ? 80 : 48,
            left: 16,
            child: _InfoToggle(
              isExpanded: _showInfo,
              onToggle: () => setState(() => _showInfo = !_showInfo),
            ),
          ),
        if (_showInfo)
          Positioned(
            top: 96,
            left: 16,
            right: 16,
            child: HUD(
              wsStatus: _wsService.status,
              processTimeMs: _wsService.processTimeMs > 0 ? _wsService.processTimeMs : null,
              backendUrl: _apiBaseUrl,
            ),
          ),
        if (!_isRecording && !_isCountdownActive)
          Positioned(
            bottom: 48,
            left: 0,
            right: 0,
            child: Column(
              children: [
                _ModelToggle(
                  selectedModel: _selectedModel,
                  onModelChanged: (model) {
                    _onModelChanged(model);
                  },
                ),
                const SizedBox(height: 16),
                _RecordButton(
                  isRecording: _isRecording,
                  onTap: _toggleRecording,
                ),
              ],
            ),
          ),
        if (_isRecording && !_isCountdownActive)
          Positioned(
            bottom: 48,
            left: 0,
            right: 0,
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                _RecordButton(
                  isRecording: _isRecording,
                  onTap: _toggleRecording,
                ),
              ],
            ),
          ),
        if (_isRecording && !_isCountdownActive)
          Positioned(
            bottom: 120,
            left: 0,
            right: 0,
            child: Center(
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                decoration: BoxDecoration(
                  color: const Color(0x99000000),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text(
                  'Recording',
                  style: TextStyle(
                    color: _wsService.status == 'connected' ? const Color(0xFF00ff88) : const Color(0xFFFF4444),
                    fontSize: 12,
                    fontWeight: FontWeight.w600,
                    letterSpacing: 0.5,
                    fontFamily: '.SF Pro Text',
                  ),
                ),
              ),
            ),
          ),
      ],
    );
  }
}

class _RecordButton extends StatelessWidget {
  final bool isRecording;
  final VoidCallback onTap;

  const _RecordButton({
    required this.isRecording,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        width: 80,
        height: 80,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          color: isRecording ? Colors.white : const Color(0xFFFF3B30),
          border: Border.all(color: Colors.white, width: 4),
        ),
        child: Center(
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 200),
            width: isRecording ? 28 : 0,
            height: isRecording ? 28 : 0,
            decoration: const BoxDecoration(
              color: Color(0xFFFF3B30),
              borderRadius: BorderRadius.all(Radius.circular(4)),
            ),
          ),
        ),
      ),
    );
  }
}

class _ModelToggle extends StatelessWidget {
  final String selectedModel;
  final ValueChanged<String> onModelChanged;

  const _ModelToggle({
    required this.selectedModel,
    required this.onModelChanged,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: const Color(0x99000000),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          _ModelChip(
            label: 'Custom',
            isSelected: selectedModel == 'partlitunet',
            onTap: () => onModelChanged('partlitunet'),
          ),
          const SizedBox(width: 8),
          _ModelChip(
            label: 'YOLO',
            isSelected: selectedModel == 'yolo',
            onTap: () => onModelChanged('yolo'),
          ),
        ],
      ),
    );
  }
}

class _ModelChip extends StatelessWidget {
  final String label;
  final bool isSelected;
  final VoidCallback onTap;

  const _ModelChip({
    required this.label,
    required this.isSelected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
        decoration: BoxDecoration(
          color: isSelected ? const Color(0xFF0066cc) : const Color(0x33000000),
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
            color: isSelected ? const Color(0xFF0066cc) : const Color(0x66ffffff),
            width: 1,
          ),
        ),
        child: Text(
          label,
          style: TextStyle(
            color: isSelected ? Colors.white : const Color(0xFFcccccc),
            fontSize: 12,
            fontWeight: FontWeight.w600,
            fontFamily: '.SF Pro Text',
          ),
        ),
      ),
    );
  }
}

class _CountdownOverlay extends StatelessWidget {
  final int count;

  const _CountdownOverlay({required this.count});

  @override
  Widget build(BuildContext context) {
    return Container(
      color: const Color(0x66000000),
      child: Center(
        child: AnimatedScale(
          scale: 1.0,
          duration: const Duration(milliseconds: 300),
          child: Text(
            '$count',
            style: const TextStyle(
              color: Colors.white,
              fontSize: 120,
              fontWeight: FontWeight.w300,
              fontFamily: '.SF Pro Display',
              letterSpacing: -2,
            ),
          ),
        ),
      ),
    );
  }
}

class _InfoToggle extends StatelessWidget {
  final bool isExpanded;
  final VoidCallback onToggle;

  const _InfoToggle({
    required this.isExpanded,
    required this.onToggle,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onToggle,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        decoration: BoxDecoration(
          color: const Color(0x99000000),
          borderRadius: BorderRadius.circular(20),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              isExpanded ? Icons.expand_more : Icons.info_outline_rounded,
              color: Colors.white,
              size: 18,
            ),
            const SizedBox(width: 6),
            Text(
              isExpanded ? 'Hide Info' : 'Info',
              style: const TextStyle(
                color: Colors.white,
                fontSize: 12,
                fontWeight: FontWeight.w600,
                fontFamily: '.SF Pro Text',
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ConnectingIndicator extends StatelessWidget {
  const _ConnectingIndicator();

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: const Color(0x99000000),
        borderRadius: BorderRadius.circular(20),
      ),
      child: const Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          SizedBox(
            width: 12,
            height: 12,
            child: CircularProgressIndicator(
              strokeWidth: 2,
              color: Color(0xFF0066cc),
            ),
          ),
          SizedBox(width: 8),
          Text(
            'Connecting...',
            style: TextStyle(
              color: Colors.white,
              fontSize: 12,
              fontWeight: FontWeight.w600,
              fontFamily: '.SF Pro Text',
            ),
          ),
        ],
      ),
    );
  }
}
