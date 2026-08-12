export default class FrameEncoder {
  private targetWidth: number;
  private quality: number;

  constructor(targetWidth: number = 640, quality: number = 0.5) {
    this.targetWidth = targetWidth;
    this.quality = quality;
  }

  getTargetWidth(): number {
    return this.targetWidth;
  }

  getQuality(): number {
    return this.quality;
  }
}
