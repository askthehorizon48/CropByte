import tensorflow as tf
import numpy as np

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="cropbyte_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare a sample input (adjust shape and values as needed)
input_shape = input_details[0]['shape']
sample_input = np.array([[0.5, 0.3, 0.7]], dtype=np.float32)  # Example input

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], sample_input)

# Run inference
interpreter.invoke()

# Get the output
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Prediction:", output_data)
