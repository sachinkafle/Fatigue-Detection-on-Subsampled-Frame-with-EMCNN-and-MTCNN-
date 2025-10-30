import os 
import numpy as np
import cv2
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

class Dataset:
    categories = ['eye_open', 'eye_close', 'mouth_open', 'mouth_close']
    data_dir = './datasets'

    def __init__(self):
        self.X = []
        self.y = []
    
    def prepare_dataset(self):
        for category in self.categories:
            path = os.path.join(self.data_dir, category)
            class_num = self.categories.index(category)
            for img_name in os.listdir(path):
                try:
                    img_path = os.path.join(path, img_name)
                    img = cv2.imread(img_path)
                    img_resized = cv2.resize(img, (175,175))
                    self.X.append(img_resized)
                    self.y.append(class_num)
                except Exception as e:
                    print(f"error loading image {img_name}: {e}")
                
        self.X = np.array(self.X) / 255.0
        self.y = to_categorical(self.y, num_classes=len(self.categories))

    def get_train_test_split(self, test_size=0.2):
        self.prepare_dataset()
        return train_test_split(self.X, self.y, test_size=test_size, random_state=42)

def build_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(175,175,3)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(4, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    # prepare the dataset
    dataset = Dataset()
    X_train, X_test, y_train, y_test = dataset.get_train_test_split(test_size=0.2)
    # build and train the cnn model
    model = build_cnn_model()
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
    # save the trained model
    model.save("./models/fatigue_model.h5")
    print("Model training completed and saved as './models/fatigue_model.h5'")

if __name__ == '__main__':
    train_model()
