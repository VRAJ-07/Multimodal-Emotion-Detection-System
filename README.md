# Multimodal Emotion Detection System

## Overview
This **Multimodal Emotion Detection System** combines **speech emotion recognition** and **facial expression recognition** to predict human emotions from both audio and image inputs. The system uses pre-trained deep learning models to identify emotions such as Angry, Happy, Sad, Fear, Neutral, and more by analyzing speech data (from audio files) and facial expressions (from images).

By utilizing both modalities (speech and facial expression), this system can enhance emotion detection accuracy and provide a more comprehensive understanding of the subject's emotional state.

## How It Works
1. **Speech Emotion Recognition**:
   - Uses the **MFCC** (Mel-frequency cepstral coefficients) features extracted from audio signals to predict the emotion using a pre-trained speech emotion model.
   
2. **Facial Expression Recognition**:
   - Uses a pre-trained **convolutional neural network (CNN)** to predict emotions based on grayscale facial images.

3. **Multimodal Emotion Detection**:
   - Combines the predictions from both speech and facial models to present the detected emotions from both modalities.

## Features
- **Real-time Emotion Detection**: Detect emotions from real-time or pre-recorded audio and images.
- **Multimodal Approach**: Combines results from both speech and facial expressions to improve overall emotion detection accuracy.
- **Visualization**: Displays the input image and plays the audio used for emotion detection.

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/VRAJ-07/Realtime_Emotion_-_Speech_Detection.git
cd Multimodal_Emotion_Detection
```

### 2. Install dependencies
Make sure you have the following Python libraries installed:
- `librosa` (for audio feature extraction)
- `numpy` (for numerical computations)
- `pandas` (for data manipulation)
- `matplotlib` (for visualizations)
- `seaborn` (for enhanced data visualizations)
- `tensorflow` / `keras` (for loading pre-trained models)
- `IPython.display` (for displaying audio)
- `opencv-python` (for image processing)
  
You can install all dependencies using the following command:
```bash
pip install librosa numpy pandas matplotlib seaborn tensorflow opencv-python
```

### 3. Download Pre-trained Models
- Place the pre-trained speech emotion recognition model (`emotion_model.h5`) and facial expression recognition model (`model.h5`) in the project directory.

### 4. Run the Multimodal Emotion Detection
You can now run the multimodal emotion detection script:
```bash
python multimodal_emotion_detection.py
```

## How to Use

### Functionality
- **Speech Emotion Recognition**: This function extracts audio features from an audio file and predicts the corresponding emotion using a pre-trained speech emotion model.
- **Facial Expression Recognition**: The system takes an image of a face, processes it, and predicts the emotion using a pre-trained facial expression model.
- **Multimodal Detection**: Both the speech and facial expression results are displayed together, providing a multimodal emotion analysis.

### Example Usage
To use the system, you need to provide an audio file (e.g., `happy.wav`) and an image file (e.g., `happy.jpg`). The system will predict the emotions from both the speech and the facial expression and display the results.

```python
# Example Usage
audio_path = 'audio_samples/happy.wav'
image_path = 'image_samples/happy.jpg'
multimodal_emotion_detection(audio_path, image_path)
```

This example will output:
```
Predicted Speech Emotion: Happy
Predicted Facial Expression Emotion: Happy

Speech Emotion Audio:
<Plays the audio>

Facial Expression Image:
<Displays the image>
```

### Testing with Different Emotions
You can test the system with different audio and image samples representing various emotions, such as:
- `happy.wav` and `happy.jpg`
- `sad.wav` and `sad.jpg`
- `angry.wav` and `angry.jpg`
- `fear.wav` and `fear.jpg`

Each time, the system will predict the emotions based on both speech and facial expressions.

## Pre-Trained Models
### 1. **Speech Emotion Recognition Model**
- This model is trained to classify different emotions from speech data using features such as MFCC (Mel-frequency cepstral coefficients).
  
### 2. **Facial Expression Recognition Model**
- A CNN-based model that classifies emotions from facial images, using grayscale images of size 48x48 pixels as input.

## Emotion Mapping
The system uses the following emotion labels for both speech and facial emotion recognition:

```
0: Angry
1: Disgust
2: Fear
3: Happy
4: Neutral
5: Sad
6: Surprise
```

## Future Improvements
- **Fusion of Modalities**: Implement a strategy to combine speech and facial emotion predictions to give a final combined emotion result.
- **Real-Time Emotion Detection**: Extend the system to work with real-time camera and microphone inputs.
- **Support for More Emotions**: Add more emotion categories like "Boredom" or "Confusion" for better coverage.

## License
This project is licensed under the MIT License.

---

**Contributors**:  
- Your Name (your.email@example.com)  

Feel free to raise issues or contribute to the project!
