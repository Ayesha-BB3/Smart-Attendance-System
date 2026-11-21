import tensorflow as tf
from tensorflow.keras import layers, Model

def build_embedder(input_shape=(100, 100, 3), embedding_dim=128):
    inputs = layers.Input(shape=input_shape, name='input_face')
    
    x = layers.Conv2D(32, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x) 
    
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x) 
    
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x) 
    
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x) 
    
    # Yahan Flatten ki jagah GlobalAveragePooling lagaya hai
    x = layers.GlobalAveragePooling2D()(x)
    
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.4)(x) # Dropout barha diya taake yaad na kare
    x = layers.Dense(embedding_dim, name='embedding')(x)
    
    # L2 Normalize
    outputs = layers.UnitNormalization(axis=1, name='l2_norm')(x)
    
    return Model(inputs, outputs, name='face_embedder')