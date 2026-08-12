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

  encodeJpegFromFrame(frame: any): Uint8Array | null {
    if (!frame || typeof frame.toJPEG !== 'function') {
      return null;
    }
    return frame.toJPEG({ quality: this.quality });
  }

  encodeJpegFromRGBA(
    rgba: Uint8Array,
    width: number,
    height: number,
  ): Uint8Array | null {
    if (typeof global.Compression !== 'undefined') {
      const bmp = global.Compression.BMP.fromBuffer(rgba, width, height);
      return global.Compression.JPEG(bmp, this.quality);
    }
    return null;
  }
}
