# DeepHistFaces

DeepHistFaces is a hybrid face recognition system combining traditional (LBPH) and deep learning (CNN) techniques to achieve robust and accurate face recognition. It is designed to compare the performance of both methods and provide insights into their strengths and limitations.

# Features

1.Face image collection module using OpenCV.

2.Training module for LBPH and CNN models.

3.Comparison module to evaluate accuracy and confidence of both models.

4.Normalized output for LBPH distances and CNN probabilities.

5.Scalable and modular design.

# Requirements

-> Python 3.8.20

-> Virtual environment (recommended)

-> Libraries:
      1. TensorFlow
      2. OpenCV
      3. Numpy
      4. Matplotlib
      5. Git LFS ( for large file storage )

-> Hardware:
      1. Webcam (for image collection)

# Installation

-> Clone the repository:
      git clone https://github.com/Ashutosh0624/DeepHistFaces.git
      cd DeepHistFaces
->Create and activate a virtual environment:
      python -m venv MyprojEnv
      source MyprojEnv/bin/activate  # For Linux/Mac
      MyprojEnv\Scripts\activate     # For Windows
->Install dependencies:
      pip install -r requirements.txt
      (Optional) Install Git LFS for handling large files:
      git lfs install
      git lfs pull

# Dataset
The project expect to organise the dataset as follows:
dataset/
├── train/
│   ├── user1/
│   │   ├── face_1.jpg
│   │   ├── face_2.jpg
│   ├── user2/
│       ├── face_1.jpg
│       ├── face_2.jpg
├── test/
    ├── user1/
        ├── face_1.jpg

# How to Use

-> Collect Images:
      python mainApp.py
      Select option 1 to collect training or test images.
->Train Models:
      python main.py
      Select option 2 to train LBPH and CNN models.
->Test Models:
      python main.py
      Select option 3 to test and compare LBPH and CNN models.
->Compare Results:
      The comparison module will provide:
      Predicted labels.
      Normalized probabilities.
      Accuracy summaries for both LBPH and CNN.

# Results

-> LBPH Confidence Range: Varies depending on the dataset (e.g., 30% to 75%).
-> CNN Confidence: Often close to 100% due to softmax normalization.

->Example Results:
      Dataset 1: LBPH (85%), CNN (97%).
      Dataset 2: LBPH (70%), CNN (95%).
->Known Issues
     CNN overconfidence due to softmax normalization.
     LBPH sensitivity to lighting and pose variations.
->Large model file (cnn_model.h5) may require Git LFS for storage.
->Future Improvements
     Add support for more recognition models (e.g., SVM, KNN).
     Improve CNN generalization with data augmentation.
     Add a web-based interface for image collection and testing

