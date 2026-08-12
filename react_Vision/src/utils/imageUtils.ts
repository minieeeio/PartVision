import { PartDetection } from '../models/DetectionModel';

const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];

export function resizeNearest(
  src: Uint8Array,
  srcW: number,
  srcH: number,
  dstW: number,
  dstH: number,
  channels: number,
): Uint8Array {
  const dst = new Uint8Array(dstW * dstH * channels);
  const xRatio = srcW / dstW;
  const yRatio = srcH / dstH;

  for (let j = 0; j < dstH; j++) {
    const ySrc = Math.min((j * yRatio) | 0, srcH - 1);
    for (let i = 0; i < dstW; i++) {
      const xSrc = Math.min((i * xRatio) | 0, srcW - 1);
      const srcIdx = (ySrc * srcW + xSrc) * channels;
      const dstIdx = (j * dstW + i) * channels;
      for (let c = 0; c < channels; c++) {
        dst[dstIdx + c] = src[srcIdx + c];
      }
    }
  }
  return dst;
}

export function prepareInput(
  rgba: Uint8Array,
  srcW: number,
  srcH: number,
  targetSize: number = 640,
): Float32Array {
  const resized = resizeNearest(rgba, srcW, srcH, targetSize, targetSize, 4);

  const input = new Float32Array(1 * 3 * targetSize * targetSize);
  const total = targetSize * targetSize;

  for (let i = 0; i < total; i++) {
    const srcIdx = i * 4;
    for (let c = 0; c < 3; c++) {
      const dstIdx = c * total + i;
      const pixelVal = resized[srcIdx + c] / 255;
      input[dstIdx] = (pixelVal - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
    }
  }

  return input;
}

export function softmax(
  logits: Float32Array,
  numClasses: number,
  spatialSize: number,
): Float32Array {
  const probs = new Float32Array(logits.length);

  for (let i = 0; i < spatialSize; i++) {
    let maxVal = -Infinity;
    for (let c = 0; c < numClasses; c++) {
      const val = logits[c * spatialSize + i];
      if (val > maxVal) maxVal = val;
    }

    let sum = 0;
    for (let c = 0; c < numClasses; c++) {
      const exp = Math.exp(logits[c * spatialSize + i] - maxVal);
      probs[c * spatialSize + i] = exp;
      sum += exp;
    }

    for (let c = 0; c < numClasses; c++) {
      probs[c * spatialSize + i] /= sum;
    }
  }

  return probs;
}

export function extractDetections(
  probs: Float32Array,
  width: number,
  height: number,
  numClasses: number,
  confidenceThreshold: number,
  classLabels: Record<number, string>,
): PartDetection[] {
  const detections: PartDetection[] = [];
  const spatialSize = width * height;

  for (let c = 1; c < numClasses; c++) {
    let maxConf = 0;
    let minX = width;
    let minY = height;
    let maxX = 0;
    let maxY = 0;
    let hasPixels = false;

    for (let i = 0; i < spatialSize; i++) {
      const val = probs[c * spatialSize + i];
      if (val > maxConf) maxConf = val;
      if (val >= confidenceThreshold) {
        hasPixels = true;
        const x = i % width;
        const y = Math.floor(i / width);
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
    }

    if (hasPixels && maxConf >= confidenceThreshold) {
      const label = classLabels[c] || `CLASS_${c}`;
      detections.push({
        label,
        confidence: maxConf,
        x_min: minX / width,
        y_min: minY / height,
        width: (maxX - minX + 1) / width,
        height: (maxY - minY + 1) / height,
      });
    }
  }

  return detections;
}
