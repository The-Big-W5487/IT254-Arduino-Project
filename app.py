import cv2
import numpy as np
import tensorflow as tf
import serial
import time

# 1. Connect to your Arduino Mega 2560 on COM3
print("Connecting to Arduino Mega...")
arduino = serial.Serial(port='COM3', baudrate=9600, timeout=1)
time.sleep(2) 
print("==================================================")
print(" STATUS: Successfully connected to Arduino Mega!")
print("==================================================")

# 2. Force Load the TensorFlow Lite Model
print("Loading TFLite Model...")
interpreter = tf.lite.Interpreter(model_path="model_unquantized.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
class_names = open("labels.txt", "r").readlines()
print(" STATUS: TFLite Model loaded successfully!")
print("==================================================")

# 3. Initialize Camera
camera = cv2.VideoCapture(0)

print("\n[SYSTEM READY]")
print("-> Hold your ID steady in front of the webcam.")
print("-> PRESS SPACEBAR TO SCAN THE ID.")
print("-> Press the 'ESC' key to close the app.\n")

while True:
    ret, image = camera.read()
    if not ret:
        print("Error: Camera feed disconnected.")
        break

    # Create a nice clean overlay guide for your demo
    display_image = image.copy()
    cv2.putText(display_image, "READY: Press [SPACEBAR] to Scan", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.imshow("ID Scanner Window", display_image)
    
    keyboard_input = cv2.waitKey(1)
    
    # --- IF SPACEBAR IS PRESSED, SCAN THE ID ---
    if keyboard_input == 32: 
        print("\n[SCANNING...] Analyzing ID profile...")
        
        # Show a quick scanning notice on screen
        scan_notice = image.copy()
        cv2.putText(scan_notice, "ANALYZING ID...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("ID Scanner Window", scan_notice)
        cv2.waitKey(500) # Quick visual pause for the effect
        
        # Prepare image canvas scaling array
        image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
        image_normalized = (image_array / 127.5) - 1

        # Run TFLite Inference
        interpreter.set_tensor(input_details[0]['index'], image_normalized)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]

        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[index]

        print(f"Result: {class_name} ({confidence_score*100:.1f}%)")
        
        if "Valid ID" in class_name:
            print(">>> ACCESS GRANTED: Sending signal to Arduino...")
            arduino.write(b'V')  
            
            # Show positive confirmation window
            result_img = image.copy()
            cv2.putText(result_img, f"ACCESS GRANTED: {class_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("ID Scanner Window", result_img)
            cv2.waitKey(1)
            time.sleep(8) # Wait for the Arduino's 7-second green light to finish
            
        elif "Invalid ID" in class_name:
            print(">>> ACCESS DENIED: Sending lockout signal to Arduino...")
            arduino.write(b'I')  
            
            # Show negative confirmation window
            result_img = image.copy()
            cv2.putText(result_img, f"ACCESS DENIED: {class_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("ID Scanner Window", result_img)
            cv2.waitKey(1)
            time.sleep(4) # Wait for the Arduino's 3-second red light lockout to finish
            
        # Flush out any backlogged camera frames that piled up during the delay
        for _ in range(10):
            camera.read()
        print("[SYSTEM READY] Standing by for next scan...\n")

    # Press 'Esc' to exit the application window instantly
    elif keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
arduino.close()
print("\n[SYSTEM CLOSED] Scanner turned off safely.")
