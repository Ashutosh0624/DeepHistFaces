import cv2
import tensorflow as tf
import os
import numpy as np

def train_lbph(data_dir="dataset/train"):
    """
    Train the LBPH model using the training dataset and save it to 'lbph_model.yml'.
    """
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    images = []
    labels = []
    label_map = {}  # Changed from list to dictionary
    label_id = 0

    for user in os.listdir(data_dir):
        user_path = os.path.join(data_dir, user)
        if os.path.isdir(user_path):
            label_map[label_id] = user  # Map the label ID to the user name
            for image_name in os.listdir(user_path):
                # Filter only image files
                if image_name.endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(user_path, image_name)
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    images.append(image)
                    labels.append(label_id)
            label_id += 1

    recognizer.train(images, np.array(labels))
    recognizer.save("lbph_model.yml")
    print("LBPH Model trained and saved.")

def train_cnn(data_dir="dataset/train", img_size=(128, 128)):
    """
    Train the CNN model using the training dataset and save it to 'cnn_model.h5'.
    """
    num_classes = len(os.listdir(data_dir))  # Count the number of classes (users)
    
    # Define CNN architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')  # Output layer with num_classes neurons
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    images, labels = [], []
    for user_id, user in enumerate(os.listdir(data_dir)):
        user_path = os.path.join(data_dir, user)
        for image_name in os.listdir(user_path):
            # Filter only image files
            if image_name.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(user_path, image_name)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, img_size)  # Resize images to the specified size
                images.append(image)
                labels.append(user_id)

    # Preprocess data for CNN
    images = np.array(images).reshape(-1, img_size[0], img_size[1], 1) / 255.0  # Normalize images
    labels = tf.keras.utils.to_categorical(labels, num_classes)  # Convert labels to one-hot encoding

    # Train the CNN model
    model.fit(images, labels, epochs=100, batch_size=32)
    model.save("cnn_model.h5")  # Save the trained model
    print("CNN Model trained and saved.")
