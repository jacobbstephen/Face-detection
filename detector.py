#using opencv
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

classifier = joblib.load('face_detector_model_svm_rbf.pkl')
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     img = cv2.resize(frame, (128, 128))
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img_flat = img_gray.flatten().reshape(1, -1)
#     prediction = classifier.predict(img_flat)
#     label = 'Human Face' if prediction[0] == 1 else 'Not Human Face'
#     cv2.putText(frame,label,(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#     cv2.imshow('Face detection', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
    
# cap.release()
# cv2.destroyAllWindows()


def predict_image(image_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read the image.")
        return

    # Resize and preprocess the image
    img_resized = cv2.resize(img, (128, 128))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_flat = img_gray.flatten().reshape(1, -1)  

    # Make a prediction
    prediction = classifier.predict(img_flat)

    # Display the result
    label = 'Human Face' if prediction[0] == 1 else 'Not Human Face'
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6, 6))
    plt.imshow(img_rgb)
    plt.title(label, fontsize=16, color='green', pad=20) 
    plt.axis('off') 
    plt.show()

for filename in os.listdir('./testdata'):
    # Path to the image you want to test
    image_path = os.path.join('./testdata', filename)

    predict_image(image_path)
