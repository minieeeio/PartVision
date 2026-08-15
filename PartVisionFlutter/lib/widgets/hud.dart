import 'package:flutter/material.dart';

class HUD extends StatelessWidget {
  final String wsStatus;
  final double? processTimeMs;
  final String? backendUrl;

  const HUD({
    super.key,
    required this.wsStatus,
    this.processTimeMs,
    this.backendUrl,
  });

  @override
  Widget build(BuildContext context) {
    final isConnected = wsStatus == 'connected';
    return Positioned(
      top: 48,
      left: 16,
      right: 16,
      child: Row(
        children: [
          _HUDItem(label: 'WS', value: wsStatus, color: isConnected ? const Color(0xFF00ff88) : const Color(0xFFFF4444)),
          if (processTimeMs != null && processTimeMs! > 0)
            _HUDItem(label: 'Latency', value: '${processTimeMs!.toInt()}ms'),
          if (backendUrl != null && backendUrl!.isNotEmpty)
            Expanded(
              child: _HUDItem(label: 'Backend', value: backendUrl!, maxLines: 1),
            ),
        ],
      ),
    );
  }
}

class _HUDItem extends StatelessWidget {
  final String label;
  final String value;
  final Color? color;
  final int? maxLines;

  const _HUDItem({
    required this.label,
    required this.value,
    this.color,
    this.maxLines,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(right: 12),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: const Color(0x99000000),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            label,
            style: const TextStyle(
              color: Color(0xFF888888),
              fontSize: 10,
              fontWeight: FontWeight.w600,
              letterSpacing: 0.5,
              fontFamily: '.SF Pro Text',
            ),
          ),
          const SizedBox(height: 2),
          Text(
            value,
            style: TextStyle(
              color: color ?? Colors.white,
              fontSize: 12,
              fontWeight: FontWeight.w500,
              fontFamily: '.SF Pro Text',
            ),
            maxLines: maxLines,
            overflow: TextOverflow.ellipsis,
          ),
        ],
      ),
    );
  }
}
