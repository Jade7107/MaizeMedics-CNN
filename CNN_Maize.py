import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, BatchNormalization,
                                   Dropout, Dense, GlobalAveragePooling2D, Input)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Configuration
BATCH_SIZE = 16
IMG_SIZE = (160, 160)  # Reduced size for faster processing
CHANNELS = 3
IMG_SHAPE = (IMG_SIZE[0], IMG_SIZE[1], CHANNELS)
BASE_PATH = "/kaggle/input/corn-or-maize-leaf-disease-dataset/data"

def create_dataframe(base_path):
    """Create DataFrame with image paths and labels."""
    images = []
    labels = []
    
    class_folders = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    for class_name in class_folders:
        class_path = os.path.join(base_path, class_name)
        class_images = os.listdir(class_path)
        
        for image_name in class_images:
            image_path = os.path.join(class_path, image_name)
            images.append(image_path)
            labels.append(class_name)
    
    df = pd.DataFrame({'image': images, 'label': labels})
    print(f"\nFound {len(df)} images across {len(df['label'].unique())} classes")
    print("\nClass distribution:")
    print(df['label'].value_counts())
    return df

def create_model(num_classes):
    """Create an efficient custom CNN model."""
    model = Sequential([
        # Input Layer
        Input(shape=IMG_SHAPE),
        
        # First Block
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Second Block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Third Block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        # Final Layers
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def main():
    print("Starting corn disease classification...")
    
    try:
        # Create DataFrame
        data = create_dataframe(BASE_PATH)
        
        # Split data
        train_df, test_df = train_test_split(data, test_size=0.2, 
                                           random_state=42, stratify=data['label'])
        
        print("\nDataset splits:")
        print(f"Training samples: {len(train_df)}")
        print(f"Test samples: {len(test_df)}")
        
        # Create data generators
        train_gen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            validation_split=0.2
        )
        
        test_gen = ImageDataGenerator(rescale=1./255)
        
        # Create datasets
        train_dataset = train_gen.flow_from_dataframe(
            train_df,
            x_col='image',
            y_col='label',
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='training'
        )
        
        valid_dataset = train_gen.flow_from_dataframe(
            train_df,
            x_col='image',
            y_col='label',
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation'
        )
        
        test_dataset = test_gen.flow_from_dataframe(
            test_df,
            x_col='image',
            y_col='label',
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        
        # Create and compile model
        model = create_model(len(train_dataset.class_indices))
        
        # Compile with higher learning rate for faster convergence
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            mode='max'
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        )
        
        # Train model
        print("\nTraining model...")
        history = model.fit(
            train_dataset,
            validation_data=valid_dataset,
            epochs=25,
            callbacks=[early_stopping, reduce_lr]
        )
        
        # Evaluate model
        print("\nEvaluating model...")
        test_loss, test_acc = model.evaluate(test_dataset)
        print(f"\nTest accuracy: {test_acc:.4f}")
        
        # Generate predictions
        predictions = model.predict(test_dataset)
        y_pred = np.argmax(predictions, axis=1)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(
            test_dataset.classes,
            y_pred,
            target_names=list(train_dataset.class_indices.keys())
        ))
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.legend()
        plt.show()
        
        # Save the model
        model.save('fast_corn_disease_model.h5')
        print("\nModel saved as 'fast_corn_disease_model.h5'")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()