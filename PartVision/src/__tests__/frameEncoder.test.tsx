const mockImage = {
  resize: jest.fn().mockReturnThis(),
  toEncodedImageData: jest.fn().mockReturnValue({
    buffer: new ArrayBuffer(100),
    width: 640,
    height: 480,
    imageFormat: 'jpg',
  }),
};

jest.mock('react-native-vision-camera', () => ({
  __esModule: true,
  HybridFrameConverter: {
    convertFrameToImage: jest.fn(() => mockImage),
  },
  useCameraDevice: jest.fn(),
  useFrameOutput: jest.fn(),
  Camera: (props: any) => null,
  Frame: {} as any,
}));

import { HybridFrameConverter } from 'react-native-vision-camera';
import { encodeFrameToJpeg } from '../utils/frameEncoder';

describe('encodeFrameToJpeg', () => {
  const mockFrame = {
    isValid: true,
    width: 1280,
    height: 720,
  };

  beforeEach(() => {
    jest.clearAllMocks();
    mockImage.resize.mockReturnThis();
    mockImage.toEncodedImageData.mockReturnValue({
      buffer: new ArrayBuffer(100),
      width: 640,
      height: 480,
      imageFormat: 'jpg',
    });
  });

  it('returns null for invalid frame', () => {
    const result = encodeFrameToJpeg({ isValid: false } as any);
    expect(result).toBeNull();
  });

  it('returns null for null frame', () => {
    const result = encodeFrameToJpeg(null as any);
    expect(result).toBeNull();
  });

  it('calls HybridFrameConverter with the frame', () => {
    encodeFrameToJpeg(mockFrame as any);
    expect(HybridFrameConverter.convertFrameToImage).toHaveBeenCalledWith(mockFrame);
  });

  it('resizes image to 640x480', () => {
    encodeFrameToJpeg(mockFrame as any);
    expect(mockImage.resize).toHaveBeenCalledWith(640, 480);
  });

  it('encodes to JPEG with quality 70', () => {
    encodeFrameToJpeg(mockFrame as any);
    expect(mockImage.toEncodedImageData).toHaveBeenCalledWith('jpg', 70);
  });

  it('returns the encoded buffer and dimensions', () => {
    const result = encodeFrameToJpeg(mockFrame as any);
    expect(result).not.toBeNull();
    expect(result?.buffer).toBeInstanceOf(ArrayBuffer);
    expect(result?.width).toBe(640);
    expect(result?.height).toBe(480);
  });
});
