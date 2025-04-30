import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, datasets, layers, models
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import cv2
import os

# Load dataset
path = os.path.dirname(os.path.realpath(__file__)) + '/images'
labels = ['tenniscourt', 'storagetanks', 'sparseresidential', 'runway', 'river', 'parkinglot', 'overpass', 'mobilehomepark', 
           'mediumresidential', 'intersection', 'harbor', 'golfcourse', 'freeway', 'forest', 'denseresidential', 'chaparral', 
           'buildings', 'beach', 'baseballdiamond' ,'airplane' , 'agricultural']

img_list = []
targets = []

for label in labels:
    label_path = os.path.join(path, label)
    count = 0

    for filename in os.listdir(label_path):
        if count >= 100:
            break
        count += 1

        img = cv2.imread(os.path.join(label_path, filename))
        img = cv2.resize(img, (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize
        img = img / 255.0
        img_list.append(img)
        targets.append(label)

img_list = np.array(img_list)
targets = np.array(targets)

# Randomize/Shuffle list
np.random.seed(0)
permuted_idx = np.random.permutation(img_list.shape[0])

# Split into 70% training and 30% testing/validation
split = int(img_list.shape[0] * 0.70)
# 15% for testing 15% for validation
val_split = int(img_list.shape[0] * 0.85)

X_train = img_list[permuted_idx[:split]]
y_train = targets[permuted_idx[:split]]

X_val = img_list[permuted_idx[split:val_split]]
y_val = targets[permuted_idx[split:val_split]]

X_test = img_list[permuted_idx[val_split:]]
y_test = targets[permuted_idx[val_split:]]

# Encode labels to integers
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.fit_transform(y_val)
y_test_encoded = label_encoder.transform(y_test)
targets_encoded = label_encoder.transform(targets)

# Convert integer labels to one-hot encoding
y_train_onehot = to_categorical(y_train_encoded, 21)
y_val_onehot = to_categorical(y_val_encoded, 21)
y_test_onehot = to_categorical(y_test_encoded, 21)

# Data augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=90,
    width_shift_range=0.3,
    height_shift_range=0.3,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

# Load pre-trained VGG16 model with no top layer
base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

# Freeze the convolutional base
for layer in base_model.layers:
    layer.trainable = False

# Add custom top layers
model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(21, activation='softmax')
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[tf.keras.metrics.CategoricalAccuracy()],
)

# Train the model with data augmentation
history = model.fit(
    datagen.flow(X_train, y_train_onehot, batch_size=32),
    epochs=20,
    validation_data=(X_val, y_val_onehot),
)

model.summary()

# Predict the labels of the test set
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

# Confusion Matrix
conf_mx = confusion_matrix(y_test_encoded, y_pred)
print(conf_mx)

accuracy = (np.trace(conf_mx) / len(y_test)) * 100
print("Accuracy =", accuracy, "%")

# Plot accuracy
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# Plot loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# Test model on entire dataset
all_pred = model.predict(img_list)
all_pred = np.argmax(all_pred, axis=1)

# Confusion Matrix
all_conf_mx = confusion_matrix(targets_encoded, all_pred)
print(all_conf_mx)

all_accuracy = (np.trace(all_conf_mx) / len(targets)) * 100
print("Accuracy =", all_accuracy, "%")