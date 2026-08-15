import 'dart:convert';
import 'package:http/http.dart' as http;

class AppConfig {
  final String apiBaseUrl;

  const AppConfig({required this.apiBaseUrl});

  factory AppConfig.fromJson(Map<String, dynamic> json) {
    return AppConfig(
      apiBaseUrl: json['api_base_url'] as String? ?? '',
    );
  }
}

class ConfigService {
  static const _remoteConfigUrl =
      'https://raw.githubusercontent.com/PrageshShrestha/hasslefree/main/partVision.json';

  static const _defaultWsPath = '/ws/segment';
  static AppConfig? _cachedConfig;

  static Future<AppConfig> fetchRemoteConfig() async {
    if (_cachedConfig != null) return _cachedConfig!;

    print('[Config] Fetching remote config from $_remoteConfigUrl');
    try {
      final response = await http.get(Uri.parse(_remoteConfigUrl)).timeout(
            const Duration(seconds: 3),
          );

      print('[Config] Response status: ${response.statusCode}');
      print('[Config] Response body: ${response.body}');

      if (response.statusCode != 200) {
        throw Exception('Failed to load config: ${response.statusCode}');
      }

      _cachedConfig = AppConfig.fromJson(jsonDecode(response.body));
      print('[Config] Parsed api_base_url: ${_cachedConfig!.apiBaseUrl}');
      return _cachedConfig!;
    } catch (e) {
      print('[Config] Fetch error: $e');
      rethrow;
    }
  }

  static String resolveWebSocketUrl(String apiBaseUrl) {
    final trimmed = apiBaseUrl.trim();
    if (trimmed.isEmpty) {
      throw ArgumentError('apiBaseUrl must not be empty');
    }

    Uri parsed;
    if (trimmed.startsWith('http://') ||
        trimmed.startsWith('https://') ||
        trimmed.startsWith('ws://') ||
        trimmed.startsWith('wss://')) {
      parsed = Uri.parse(trimmed);
    } else {
      parsed = Uri.parse('http://$trimmed');
    }

    final scheme = parsed.scheme.isEmpty ? 'http' : parsed.scheme;
    final host = parsed.host;
    final port = parsed.hasPort ? ':${parsed.port}' : '';
    final path = parsed.path;

    if (host.isEmpty) {
      throw ArgumentError('Invalid apiBaseUrl: no host found in "$trimmed"');
    }

    String wsScheme;
    if (scheme == 'https' || scheme == 'wss') {
      wsScheme = 'wss';
    } else {
      wsScheme = 'ws';
    }

    final base = '$wsScheme://$host$port';
    final normalizedPath = path.replaceAll(RegExp(r'/+$'), '');
    final needsSegmentPath = !normalizedPath.endsWith(_defaultWsPath);

    if (needsSegmentPath) {
      final cleanPath = normalizedPath.isEmpty ? '' : '$normalizedPath';
      return '$base$cleanPath$_defaultWsPath';
    }
    return '$base$normalizedPath';
  }

  static String? getFallbackBaseUrl() {
    return null;
  }
}
