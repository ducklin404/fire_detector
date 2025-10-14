from fire_logistic_tinyml.config import *
from fire_logistic_tinyml.data_loader import load_data, split_data
from fire_logistic_tinyml.sklearn_trainer import train_sklearn
from fire_logistic_tinyml.keras_builder import build_keras_model, transfer_weights, evaluate_keras
from fire_logistic_tinyml.tflite_convert import convert_to_tflite
from fire_logistic_tinyml.header_generator import tflite_to_c_header

def main():
    # 1. Load data
    X, y = load_data(CSV_PATH)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 2. Train sklearn model
    sk_model, scaler, acc_sk = train_sklearn(X_train, X_test, y_train, y_test)

    # 3. Build Keras equivalent
    keras_model, normalizer = build_keras_model(X_train, X.shape[1])

    # 4. Transfer weights
    transfer_weights(keras_model, sk_model)

    # 5. Evaluate
    acc_keras = evaluate_keras(keras_model, X_test, y_test)

    # 6. Save model
    keras_model.save(KERAS_H5)
    print("Saved Keras model to", KERAS_H5)

    # 7. Convert to TFLite and quantize
    tflite_int8_model = convert_to_tflite(keras_model, TFLITE_FP32, TFLITE_INT8, X_train)

    # 8. Generate C header
    tflite_to_c_header(TFLITE_INT8, HEADER_INT8, array_name="fire_logistic_int8")
    print("All done. Header ready for TFLite Micro use.")

if __name__ == "__main__":
    main()
