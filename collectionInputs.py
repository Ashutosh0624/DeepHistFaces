import cv2
import os

def collect_images(user_name, data_type ="train", image_count=100):
    base_dir = "dataset"
    output_dir = os.path.join(base_dir, data_type, user_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory:{output_dir}")
    
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            count += 1
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (128, 128))
            cv2.imwrite(f"{output_dir}/face_{count}.jpg", face_resized)
            print(f"Saved: {output_dir}/face_{count}.jpg")

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Collecting Images", frame)

        if count >= image_count or cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Image collection for {user_name} in {data_type} set complete.")