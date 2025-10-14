import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score

def build_keras_model(X_train, n_features):
    normalizer = tf.keras.layers.Normalization(axis=-1, dtype=tf.float32)
    normalizer.adapt(X_train)

    inputs = tf.keras.Input(shape=(n_features,), dtype=tf.float32, name="input")
    x = normalizer(inputs)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='prob')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model, normalizer

def transfer_weights(keras_model, sk_model):
    sk_w = sk_model.coef_.reshape(-1, 1).astype(np.float32)
    sk_b = sk_model.intercept_.astype(np.float32)
    keras_model.get_layer('prob').set_weights([sk_w, sk_b])

def evaluate_keras(keras_model, X_test, y_test):
    y_pred_prob = keras_model.predict(X_test, batch_size=64)
    y_pred = (y_pred_prob.ravel() >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    print("Keras model test accuracy:", acc)
    return acc