import tensorflow as tf
from tensorflow import keras
import numpy as np

# ORIGINAL BUGGY CODE (commented out)
'''
# Bug 1: Incorrect import
import tensorflow as keras

# Bug 2: Wrong data loading
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Bug 3: Missing preprocessing
X_train = X_train.reshape(-1, 28, 28)
X_test = X_test.reshape(-1, 28, 28)

# Bug 4: Incorrect model architecture
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(64, activation='sigmoid'),  # Wrong activation
    keras.layers.Dense(10, activation='softmax')
])

# Bug 5: Wrong loss function
model.compile(optimizer='adam', 
              loss='binary_crossentropy',  # Wrong for multi-class
              metrics=['accuracy'])

# Bug 6: Incorrect training
model.fit(X_train, y_train, epochs=5)  # Missing validation, batch_size
'''

# FIXED CODE
print("🐛 Debugging and Fixing TensorFlow Code...")

# Fix 1: Correct imports
from tensorflow import keras
from tensorflow.keras import layers

# Fix 2: Correct data loading
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Fix 3: Proper preprocessing
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
X_train = X_train.reshape(-1, 28, 28, 1)  # Add channel dimension for CNN
X_test = X_test.reshape(-1, 28, 28, 1)

# Fix 4: Convert labels to categorical
y_train_categorical = keras.utils.to_categorical(y_train, 10)
y_test_categorical = keras.utils.to_categorical(y_test, 10)

# Fix 5: Correct model architecture
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')  # Correct for 10 classes
])

# Fix 6: Correct loss function
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Correct for multi-class
              metrics=['accuracy'])

print("✅ Model compiled successfully!")
model.summary()

# Fix 7: Proper training with validation
history = model.fit(X_train, y_train_categorical,
                    batch_size=128,
                    epochs=5,
                    validation_split=0.2,
                    verbose=1)

# Evaluate fixed model
test_loss, test_accuracy = model.evaluate(X_test, y_test_categorical, verbose=0)
print(f"🎯 Fixed Model Test Accuracy: {test_accuracy:.2%}")

# Save the model for use in the Streamlit app
model.save('mnist_cnn_model.h5')
print("💾 Model saved to mnist_cnn_model.h5")

print("✅ All bugs fixed! Model training completed successfully.")