import { InferenceSession, Tensor } from 'onnxruntime-react-native';
import { prepareInput, softmax, extractDetections } from '../utils/imageUtils';
import { PartDetection } from '../models/DetectionModel';
import { INFERENCE_CONFIG, CLASS_LABELS } from '../utils/constants';

export class InferenceManager {
  private session: InferenceSession | null = null;
  private loaded = false;
  private inputName: string = '';
  private outputName: string = '';
  private modelPath: number | string;

  constructor(modelPath?: number | string) {
    this.modelPath = modelPath ?? INFERENCE_CONFIG.MODEL_PATH;
  }

  async load(): Promise<boolean> {
    try {
      this.session = await InferenceSession.create(this.modelPath, {
        executionProviders: [INFERENCE_CONFIG.EXECUTION_PROVIDER],
      });

      const inputs = this.session.inputNames;
      const outputs = this.session.outputNames;

      this.inputName = inputs[0] || 'input';
      this.outputName = outputs[0] || 'output';

      this.loaded = true;
      console.log('[InferenceManager] Model loaded:', {
        inputs,
        outputs,
      });
      return true;
    } catch (e) {
      console.error('[InferenceManager] Failed to load model:', e);
      this.loaded = false;
      return false;
    }
  }

  async infer(
    rgba: Uint8Array,
    width: number,
    height: number,
  ): Promise<PartDetection[]> {
    if (!this.session || !this.loaded) {
      return [];
    }

    const inputTensor = prepareInput(rgba, width, height, INFERENCE_CONFIG.TARGET_SIZE);

    const feeds: Record<string, Tensor> = {
      [this.inputName]: new Tensor(
        'float32',
        inputTensor,
        [1, 3, INFERENCE_CONFIG.TARGET_SIZE, INFERENCE_CONFIG.TARGET_SIZE],
      ),
    };

    const results = await this.session.run(feeds);
    const output = results[this.outputName];

    const logits = output.data as Float32Array;
    const dims = output.dims as number[];
    // dims typically: [1, numClasses, H, W]
    const numClasses = dims[1] || INFERENCE_CONFIG.NUM_CLASSES;
    const outH = dims[2] || INFERENCE_CONFIG.TARGET_SIZE;
    const outW = dims[3] || INFERENCE_CONFIG.TARGET_SIZE;
    const spatialSize = outH * outW;

    const probs = softmax(logits, numClasses, spatialSize);

    return extractDetections(
      probs,
      outW,
      outH,
      numClasses,
      INFERENCE_CONFIG.CONFIDENCE_THRESHOLD,
      CLASS_LABELS,
    );
  }

  isLoaded(): boolean {
    return this.loaded;
  }

  async dispose(): Promise<void> {
    if (this.session) {
      await this.session.release();
      this.session = null;
    }
    this.loaded = false;
  }
}

export const inferenceManager = new InferenceManager();
