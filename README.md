# Deepfake Interaction System

**CENG 384 - Intro. to Signal Processing**

A desktop application that demonstrates Digital Signal Processing (DSP) concepts through voice cloning and real-time face swapping.

## Team Members

- Alperen Gokay Goktas
- Mohammed Alweli
- Asli Sila Caralan
- Nil Savci

## Features

### Voice Cloning
- 8 preset neural voice styles (Microsoft Edge TTS)
- Real voice cloning from audio samples using f5-tts-mlx (Apple Silicon MLX)
- Interactive waveform timeline with click-to-preview and region selection
- Automatic noise reduction (spectral gating)
- Automatic speech transcription (Whisper MLX)
- Pitch and speed adjustment
- Quality metrics: SNR, Audio Clarity, Voice Match (MFCC cosine similarity), Pitch Accuracy
- Visualizations: spectrograms, waveform comparison, frequency spectrum analysis

### Face Swap
- Live webcam face swap using InsightFace + inswapper_128 neural model
- CoreML GPU acceleration on Apple Silicon
- Face detection with bounding box and confidence percentage
- 5-point facial landmark tracking
- Expression detection (head direction, mouth state, eye state)
- Expanded blending for smooth forehead/hairline transitions
- Asynchronous processing for smooth webcam display
- Quality metrics: PSNR, SSIM, Face Detection confidence, Blend Quality
- Analysis visualizations: difference heatmap, face region comparison

### Logs
- Real-time color-coded log viewer (info, success, error, warning)
- Tracks all operations with timestamps

## DSP Techniques Demonstrated

**Audio DSP:**
- Fast Fourier Transform (FFT) for frequency analysis
- Mel-frequency spectrogram and MFCC extraction
- Pitch detection (pYIN algorithm)
- Pitch shifting and time stretching
- Audio resampling and normalization
- Spectral gating noise reduction
- Butterworth filter design

**Image/Video DSP:**
- Face detection and landmark extraction
- Gaussian blur for mask feathering
- Alpha blending for seamless compositing
- Image difference computation
- Color space conversion

## Tech Stack

| Category | Libraries |
|----------|-----------|
| Audio Processing | librosa, soundfile, sounddevice, noisereduce |
| Voice Cloning | f5-tts-mlx, edge-tts, mlx-whisper |
| Computer Vision | OpenCV, MediaPipe, InsightFace |
| ML Inference | ONNX Runtime (CoreML), MLX |
| UI | CustomTkinter, matplotlib |
| Evaluation | numpy, scipy |

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd "dsp project"

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install additional packages
pip install edge-tts f5-tts-mlx mlx-whisper noisereduce sounddevice
pip install insightface onnxruntime matplotlib
```

### Model Downloads

The following models are downloaded automatically on first run:
- **InsightFace buffalo_l** (~280MB) - Face detection and analysis
- **f5-tts-mlx** (~500MB) - Voice cloning model
- **Whisper** (~500MB) - Speech transcription

The **inswapper_128.onnx** model (529MB) needs to be placed in the `models/` directory.

## Usage

```bash
source venv/bin/activate
python main.py
```

### Voice Cloning
1. Go to **Voice Cloning** tab
2. Choose **Preset Voices** for quick generation, or **Upload Sample** for voice cloning
3. For cloning: upload audio, select a clean speech region, click **Use This Region**
4. Type text and click **Generate & Play**

### Face Swap
1. Go to **Face Swap** tab
2. Click **Upload Face Image** with a clear face photo
3. Click **Start Webcam**
4. Click **Capture Screenshot** for analysis with metrics and visualizations

## Evaluation Metrics

| Metric | What it measures | Good value |
|--------|-----------------|------------|
| SNR | Audio signal quality (dB) | > 20 dB |
| PSNR | Image quality (dB) | > 30 dB |
| SSIM | Structural similarity (0-1) | > 0.8 |
| MFCC Similarity | Voice match (0-1) | > 0.85 |

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.13
- Webcam (for face swap)
- Internet connection (for edge-tts preset voices and first-time model downloads)

## Project Structure

```
dsp project/
  main.py                    - Entry point
  config.py                  - Configuration
  requirements.txt           - Dependencies
  models/                    - AI model files
  outputs/                   - Generated outputs
  app/
    modules/
      voice_cloning.py       - Voice cloning engine
      face_swap.py           - Face swap engine
      input_module.py        - File validation
      preprocessing.py       - Audio/image preprocessing
      evaluation.py          - Quality metrics
      output_module.py       - File export
    ui/
      main_window.py         - Main window
      voice_panel.py         - Voice cloning UI
      face_panel.py          - Face swap UI
      logs_panel.py          - Log viewer
    utils/
      file_utils.py          - File utilities
      logger.py              - Logging system
```

## License

This project is for educational purposes as part of CENG 384 coursework.
