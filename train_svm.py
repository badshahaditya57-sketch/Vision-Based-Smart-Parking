import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

def extract_features(img_path):
    # Read as grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
        
    # Extract HOG features (edges and shapes)
    # These parameters determine how detailed the shape extraction is
    features = hog(img, orientations=9, pixels_per_cell=(8, 8), 
                   cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
    return features

data_dir = './processed_data'
categories = ['empty', 'occupied']
features_list = []
labels_list = []

# Limit the number of images to prevent RAM crashes and speed up training
max_images_per_class = 2000 

print("Loading images and extracting HOG features. Please wait...")

# 1. Load the processed data
for label_idx, category in enumerate(categories):
    folder_path = os.path.join(data_dir, category)
    if not os.path.exists(folder_path):
        print(f"Warning: Could not find folder {folder_path}")
        continue
        
    print(f"Processing category: {category}")
    
    # Loop through the files with a cap
    for i, file in enumerate(os.listdir(folder_path)):
        if i >= max_images_per_class:
            break
            
        img_path = os.path.join(folder_path, file)
        hog_features = extract_features(img_path)
        
        if hog_features is not None:
            features_list.append(hog_features)
            labels_list.append(label_idx)
            
        # Print a progress update every 500 images
        if (i + 1) % 500 == 0:
            print(f"  Extracted {i + 1} / {max_images_per_class} images...")

X = np.array(features_list)
y = np.array(labels_list)

print(f"\nSuccessfully extracted features from {len(X)} total parking spots.")
print("Training the SVM model...")

# 2. Split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train the model
# A linear kernel is highly efficient and works exceptionally well for HOG vectors
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)

# 4. Evaluate performance
print("\nEvaluating model accuracy on test data...")
predictions = svm_model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"Final Model Accuracy: {accuracy * 100:.2f}%\n")
print("Detailed Report:")
print(classification_report(y_test, predictions, target_names=categories))

# 5. Save the trained model
joblib.dump(svm_model, 'hog_svm_parking.pkl')
print("\nSuccess: Model saved as 'hog_svm_parking.pkl'")