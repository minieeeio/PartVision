import FrameEncoder from './FrameEncoder';

export class CameraManager {
  readonly encoder: FrameEncoder;

  constructor(targetWidth: number = 640, quality: number = 0.5) {
    this.encoder = new FrameEncoder(targetWidth, quality);
  }

  getTargetWidth(): number {
    return this.encoder.getTargetWidth();
  }

  getQuality(): number {
    return this.encoder.getQuality();
  }
}
