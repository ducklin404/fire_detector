import os

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = "1000_data_cleaned.csv"
KERAS_H5 = os.path.join(OUTPUT_DIR, "fire_logistic.h5")
TFLITE_FP32 = os.path.join(OUTPUT_DIR, "fire_logistic_fp32.tflite")
TFLITE_INT8 = os.path.join(OUTPUT_DIR, "fire_logistic_int8.tflite")
HEADER_INT8 = os.path.join(OUTPUT_DIR, "fire_logistic_int8.h")