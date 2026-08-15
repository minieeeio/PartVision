import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'screens/camera_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setEnabledSystemUIMode(SystemUiMode.immersiveSticky);
  runApp(const PartVisionApp());
}

class PartVisionApp extends StatelessWidget {
  const PartVisionApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'PartVision',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: Colors.black,
        colorScheme: const ColorScheme.dark(primary: Color(0xFF0066cc)),
        textTheme: const TextTheme(
          displayLarge: TextStyle(fontFamily: '.SF Pro Display', fontSize: 40, fontWeight: FontWeight.w600, height: 1.1, letterSpacing: 0),
          displayMedium: TextStyle(fontFamily: '.SF Pro Text', fontSize: 34, fontWeight: FontWeight.w600, height: 1.47, letterSpacing: -0.374),
          bodyLarge: TextStyle(fontFamily: '.SF Pro Text', fontSize: 17, fontWeight: FontWeight.w400, height: 1.47, letterSpacing: -0.374),
          bodyMedium: TextStyle(fontFamily: '.SF Pro Text', fontSize: 14, fontWeight: FontWeight.w400, height: 1.43, letterSpacing: -0.224),
          labelLarge: TextStyle(fontFamily: '.SF Pro Text', fontSize: 17, fontWeight: FontWeight.w400, height: 1.47, letterSpacing: -0.374),
        ),
      ),
      home: const CameraScreen(),
    );
  }
}
