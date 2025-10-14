import tensorflow as tf
import numpy as np

def convert_to_tflite(keras_model, fp32_path, int8_path, X_train):
    # Float32 model
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()
    open(fp32_path, "wb").write(tflite_model)
    print("Saved float32 TFLite:", fp32_path)

    # Representative dataset
    def representative_data_gen():
        for i in range(min(200, X_train.shape[0])):
            yield [np.expand_dims(X_train[i].astype(np.float32), axis=0)]

    # Int8 quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_int8_model = converter.convert()
    open(int8_path, "wb").write(tflite_int8_model)
    print("Saved int8-quantized TFLite:", int8_path)
    return tflite_int8_model