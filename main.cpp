#include <Arduino.h>
#include "fire_logistic_int8.h" // Generated from TFLite int8 conversion

#include "TensorFlowLite.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// ---------------------------
// Quantization parameters from your TFLite model
// ---------------------------
const float INPUT_SCALE = 14.815686225891113f;
const int INPUT_ZERO_POINT = -128;
const float OUTPUT_SCALE = 0.00390625f;
const int OUTPUT_ZERO_POINT = -128;

// Model data
extern const unsigned char fire_logistic_int8[];
extern const unsigned int fire_logistic_int8_len;

// TensorFlow Lite Micro globals
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::AllOpsResolver resolver;
constexpr int kTensorArenaSize = 20 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// ---------------------------
// Helper functions for quantization
// ---------------------------
int8_t float_to_quant_int8(float real_val, float scale, int zero_point) {
  int32_t q = (int32_t)round(real_val / scale) + zero_point;
  if (q > 127) q = 127;
  if (q < -128) q = -128;
  return (int8_t)q;
}

float dequant_int8_to_float(int8_t q_val, float scale, int zero_point) {
  return ((float)q_val - (float)zero_point) * scale;
}

// Replace with actual sensor read code
float read_sensor_temperature() { return 25.0f; }
float read_sensor_humidity() { return 50.0f; }
float read_sensor_gasAnalog() { return 0.12f; }
float read_sensor_flameDetected() { return 0.0f; }

// ---------------------------
void setup() {
  Serial.begin(9600);
  delay(1000);
  Serial.println("Starting TinyML on ESP32...");

  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(fire_logistic_int8);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch!");
    while (1) {}
  }

  interpreter = new tflite::MicroInterpreter(model, resolver, tensor_arena, kTensorArenaSize, error_reporter);

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while (1) {}
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
}

void loop() {
  float temp = read_sensor_temperature();
  float hum = read_sensor_humidity();
  float gas = read_sensor_gasAnalog();
  float flame = read_sensor_flameDetected();

  float input_vals[4] = { temp, hum, gas, flame };

  // Quantize input
  for (int i = 0; i < 4; ++i) {
    input->data.int8[i] = float_to_quant_int8(input_vals[i], INPUT_SCALE, INPUT_ZERO_POINT);
  }

  // Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed!");
    delay(1000);
    return;
  }

  int8_t q_out = output->data.int8[0];
  float prob = dequant_int8_to_float(q_out, OUTPUT_SCALE, OUTPUT_ZERO_POINT);

  Serial.print("Fire probability: ");
  Serial.println(prob, 4);
  Serial.print("Prediction: ");
  Serial.println((prob >= 0.5f) ? 1 : 0);

  delay(1000);
}
