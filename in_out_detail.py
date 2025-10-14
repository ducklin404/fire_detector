import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="outputs/fire_logistic_int8.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Input scale, zero_point:", input_details[0]['quantization'])
print("Output scale, zero_point:", output_details[0]['quantization'])