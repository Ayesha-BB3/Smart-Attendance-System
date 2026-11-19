"""
Face Embedder - Siamese CNN (From Scratch)
==========================================
Custom neural network for face embedding extraction.
NO PRE-TRAINED WEIGHTS. Random initialization.
Trained using triplet loss on LFW dataset.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model


def build_embedder(input_shape=(100, 100, 3), embedding_dim=128):
    """
    Build Siamese CNN for face embedding.
    Input: 100x100x3 RGB face image
    Output: 128-dim L2-normalized embedding vector
    """
    inputs = layers.Input(shape=input_shape, name='input_face')
    
    # Block 1
    x = layers.Conv2D(32, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)  # 50x50
    
    # Block 2
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)  # 25x25
    
    # Block 3
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)  # 12x12
    
    # Block 4
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)  # 6x6
    
    # Head
    x = layers.Flatten()(x)
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(embedding_dim, name='embedding')(x)
    
    # L2 normalize: embeddings lie on unit sphere
    outputs = layers.UnitNormalization(axis=1, name='l2_norm')(x)
    
    return Model(inputs, outputs, name='face_embedder')


if __name__ == '__main__':
    model = build_embedder()
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")