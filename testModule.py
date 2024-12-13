import cv2
import tensorflow as tf
import numpy as np
import os

# Function to normalize LBPH confidence values to probabilities (0-100%)
def lbph_to_probability(lbph_confidence, min_confidence, max_confidence):
    """
    Convert LBPH confidence values to probabilities (0-100%).
    """
    if lbph_confidence > max_confidence:
        lbph_confidence = max_confidence  # Cap at max_confidence
    if lbph_confidence < min_confidence:
        lbph_confidence = min_confidence  # Cap at min_confidence

    # Invert and normalize
    probability = 1 - (lbph_confidence - min_confidence) / (max_confidence - min_confidence)
    return probability * 100  # Convert to percentage


# Function to normalize CNN outputs (already probabilities)
def cnn_to_probability(cnn_output):
    """
    Convert CNN softmax outputs to probabilities (0-100%).
    """
    return cnn_output * 100


def test_lbph(image_path, lbph_model_path="lbph_model.yml", min_confidence=20, max_confidence=100):
    """
    Test the LBPH model using a single input image and return the probability.
    """
    try:
        # Initialize and load the LBPH model
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(lbph_model_path)

        # Load and preprocess the test image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Predict the label and confidence
        label, confidence = recognizer.predict(image)
        print(f"LBPH Raw Output: Label={label}, Confidence={confidence}")

        # Normalize LBPH confidence to probability
        probability = lbph_to_probability(confidence, min_confidence, max_confidence)
        print(f"LBPH Probability: {probability:.2f}%")

        return label, probability
    except Exception as e:
        print(f"Error in LBPH testing: {e}")
        return None, None


def test_cnn(image_path, cnn_model_path="cnn_model.h5", img_size=(128, 128)):
    """
    Test the CNN model using a single input image and return the probability.
    """
    try:
        # Load the CNN model
        model = tf.keras.models.load_model(cnn_model_path)

        # Load and preprocess the test image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Resize and reshape the image to match the model's input
        image_resized = cv2.resize(image, img_size)
        image_input = np.expand_dims(image_resized, axis=-1)  # Add channel dimension for grayscale
        image_input = np.expand_dims(image_input, axis=0)  # Add batch dimension
        image_input = image_input / 255.0  # Normalize pixel values to [0, 1]

        # Predict the label
        prediction = model.predict(image_input)
        predicted_label = np.argmax(prediction)
        confidence = np.max(prediction)
        print(f"CNN Raw Output: Label={predicted_label}, Confidence={confidence}")

        # Convert confidence to percentage
        probability = cnn_to_probability(confidence)
        print(f"CNN Probability: {probability:.2f}%")

        return predicted_label, probability
    except Exception as e:
        print(f"Error in CNN testing: {e}")
        return None, None
