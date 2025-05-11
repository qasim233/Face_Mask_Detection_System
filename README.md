# Face Mask Detection System

## Overview
This project implements a real-time face mask detection system using computer vision and deep learning techniques. The system can detect whether individuals in images or video streams are wearing face masks, with applications in public health monitoring and safety compliance.

## Key Features
- Real-time face mask detection in images and video streams
- Flask-based web application with user-friendly interface
- High accuracy (88%) using MobileNetV2 transfer learning
- Robust preprocessing pipeline including histogram equalization, Gaussian blur, and Otsu thresholding
- SSD with ResNet-10 backbone for efficient face detection

## Dataset
The model was trained on the Kaggle "Face Mask Dataset":
- Binary classification (with_mask/without_mask)
- ~7,450 total images (balanced classes)
- 80% training, 20% testing split

## Technical Specifications
- **Model Architecture**: MobileNetV2 with custom classification head
- **Face Detection**: SSD with ResNet-10 backbone
- **Image Processing**:
  - Resizing to 224×224
  - Histogram equalization
  - Gaussian blur
  - Otsu thresholding
  - Morphological operations
- **Optimization**:
  - Learning rate scheduling
  - Early stopping
  - Data augmentation
  - 20% dropout

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/face_mask_detection.git
   cd face-mask-detection
   ```

2. Install dependencies:
   ```bash
   pip install opencv-python tensorflow
   ```

3. Download the pre-trained model weights and place them in the `models` directory.

## Usage
1. Run the Flask application:
   ```bash
   python3 app.py
   ```

2. Access the web interface at `http://localhost:5000`

3. Choose between:
   - Real-time webcam detection
   - Image upload processing

## Results
The system achieves:
- 88% accuracy on test set
- Real-time performance on standard hardware
- Robust detection across various lighting conditions

Sample outputs:
- Green bounding box: Mask detected
- Red bounding box: No mask detected

## Limitations
- Reduced accuracy with extreme head poses
- Performance impact with high-resolution video
- Occlusions other than masks may cause false detections

## Future Work
- Multi-class classification for different mask types
- Edge deployment optimization
- Integration with access control systems
- Mobile application development

## Contributors
- Muhammad Qasim
- Ayaan Khan
- Abubakar Nadeem
- Ahmed Mehmood

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Kaggle for the face mask dataset
- OpenCV and TensorFlow communities
- MobileNetV2 and SSD model developers
