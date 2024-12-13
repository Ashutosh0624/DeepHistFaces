import os
from collectionInputs import collect_images
from trainModule import train_lbph, train_cnn
from testModule import test_lbph, test_cnn

def main():
    print("Welcome to the Face Recognition Project!")
    print("1. Collect Images")
    print("2. Train Models")
    print("3. Test Models")
    choice = int(input("Enter your choice: "))

    if choice == 1:  # Collect Images
        user_name = input("Enter user name: ")
        data_type = input("Enter data type (train/test): ").strip().lower()
        if data_type not in ["train", "test"]:
            print("Invalid data type! Please enter 'train' or 'test'.")
            return
        image_count = int(input("How many images to collect? "))
        collect_images(user_name, data_type, image_count)

    elif choice == 2:  # Train Models
        print("Training LBPH and CNN models...")
        train_lbph()
        train_cnn()

    elif choice == 3:  # Test Models
        model_type = input("Enter model type (lbph/cnn): ").strip().lower()
        if model_type not in ["lbph", "cnn"]:
            print("Invalid model type! Please enter 'lbph' or 'cnn'.")
            return

        # Prompt for user name
        user_name = input("Enter user name (folder name in test dataset): ").strip()
        base_dir = "/home/ashutosh/Desktop/MainProject/dataset/test"
        user_dir = os.path.join(base_dir, user_name)

        # Display available images
        if os.path.exists(user_dir):
            print(f"Images available in {user_dir}:")
            available_images = [
                file for file in os.listdir(user_dir)
                if file.endswith(('.jpg', '.jpeg', '.png'))
            ]
            if not available_images:
                print("No images found in this directory!")
                return

            for file in available_images:
                print(f" - {file}")
        else:
            print(f"User directory not found: {user_dir}")
            return

        # Prompt for image name
        image_name = input("Enter image name (e.g., face_1.jpg): ").strip()

        # Construct and validate the path
        image_path = os.path.join(user_dir, image_name)
        if not os.path.exists(image_path):
            print(f"Error: Test image not found at {image_path}")
            print("Please verify the user name, file name, and folder structure.")
            return

        # Call the respective testing function
        if model_type == "lbph":
            test_lbph(image_path)
        elif model_type == "cnn":
            test_cnn(image_path)

if __name__ == "__main__":
    main()
