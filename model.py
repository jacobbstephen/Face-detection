import cv2
import os
import numpy as np
from sklearn import svm
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,  classification_report, confusion_matrix
import joblib

def load_images_from_folder(folder):
    images = []
    labels = []
    print('Image Loading....')
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        
        if img is not None:
            img = cv2.resize(img, (128, 128))
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_flat = img_gray.flatten()  
            images.append(img_flat)  
            
            if 'Human' in filename:
                labels.append(1)
            else:
                labels.append(0)
    
    return images, labels


x,y = load_images_from_folder('Dataset')
print('Image Loading Completed....')
X = np.array(x)
Y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# classifier = linear_model.LogisticRegression(max_iter=1000)
classifier = svm.SVC(kernel='rbf', C=10, gamma='scale')  
# classifier = svm.SVC(kernel='poly', degree=3, C=1.0, coef0=1)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Evaluate the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump(classifier, 'face_detector_model_svm_rbf.pkl')

    
    
    