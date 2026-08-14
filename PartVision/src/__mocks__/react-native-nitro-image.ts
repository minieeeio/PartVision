export const Images = {
  loadFromRawPixelData: jest.fn(),
};

export const NitroImage = (props: any) => null;
export type Image = any;
export type PixelFormat = 'BGRA' | 'ARGB';
export type ImageFormat = 'jpg' | 'png' | 'heic';
export interface RawPixelData {
  buffer: ArrayBuffer;
  width: number;
  height: number;
  pixelFormat: PixelFormat;
}
export interface EncodedImageData {
  buffer: ArrayBuffer;
  width: number;
  height: number;
  imageFormat: ImageFormat;
}
