import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import requests
import os

# Define the class for Apple Detection
class AppleDetectorStreamlit:
    def __init__(self):
        self.model_options = {
            'yolov8s.pt': 'YOLOv8 Small',
            'yolov8m.pt': 'YOLOv8 Medium',
            'yolov8l.pt': 'YOLOv8 Large',
            'yolov8x.pt': 'YOLOv8 Large',
            # Add more model options here
        }
        
        self.model_path = "models"
        
    def download_model(self, model_name):
        model_url = f'https://github.com/ultralytics/yolov8/releases/download/v1.0/{model_name}'
        print(f"Downloading model {model_name} from {model_url}...")
        try:
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            with open(os.path.join(self.model_path, model_name), 'wb') as model_file:
                for chunk in response.iter_content(chunk_size=8192):
                    model_file.write(chunk)
            print(f"Model {model_name} downloaded and saved to {self.model_path}.")
        except Exception as e:
            print(f"Failed to download model {model_name}: {e}")

    def detect_apples(self, image, model_name):
        try:
            model = YOLO(os.path.join(self.model_path, model_name))
            results = model.predict(image)
            print("successfully detected apples")
            return results
        except Exception as e:
            print(f"Error detecting apples: {e}")
            return None

    def process_image(self, image, model_name):
        try:
            results = self.detect_apples(image, model_name)
            if results:
                detections = results[0].boxes.data.cpu().numpy()
                detected_image, apple_count, apple_centers = self.draw_boxes(image, detections)
                return detected_image, apple_count, apple_centers
            else:
                return image, 0, []
        except Exception as e:
            print(f"Error processing image: {e}")
            return image, 0, []

    def draw_boxes(self, image, detections):
        apple_count = 0
        apple_centers = []
        for detection in detections:
            x1, y1, x2, y2, _, class_id = detection
            if int(class_id) == 47:
                apple_count += 1
                midpoint_x = int((x1 + x2) / 2)
                midpoint_y = int((y1 + y2) / 2)
                apple_centers.append((midpoint_x, midpoint_y))
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)  # Dark blue color, thickness=3
                cv2.putText(image, f"Apple: {apple_count}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
                cv2.circle(image, (midpoint_x, midpoint_y), 5, (255, 0, 0), 2)  # Blue center point
        return image, apple_count, apple_centers

    def detect_apples_streamlit(self, input_image, model_name):
        try:
            image_np = np.array(input_image)
            detected_image, apple_count, apple_centers = self.process_image(image_np, model_name)
            return detected_image, apple_count, apple_centers
        except Exception as e:
            print(f"Error processing Streamlit image: {e}")
            return None, 0, []

# Initialize the Streamlit app
app = AppleDetectorStreamlit()

# Main Streamlit app
def main():
    st.image("https://freepngimg.com/thumb/categories/1017.png", width=100)
    st.title("Apple Detection App")

    # Select detection model
    model_name = st.sidebar.selectbox("Select Detection Model", list(app.model_options.keys()))

    # Upload image
    uploaded_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if uploaded_image is not None:
        # Convert uploaded image to OpenCV format
        try:
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            input_image = cv2.imdecode(file_bytes, 1)

            # Display original image
            st.image(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)

            # Detect apples
            detected_image, apple_count, apple_centers = app.detect_apples_streamlit(input_image, model_name)

            # Display detected image, apple count, and apple midpoint coordinates
            if detected_image is not None:
                st.image(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB), caption="Detected Image", use_column_width=True)
                st.write(f"Number of apples detected: {apple_count}")
                st.write("Midpoint coordinates of detected apples:")
                for i, center in enumerate(apple_centers, 1):
                    st.write(f"Apple {i}: ({center[0]}, {center[1]})")
            else:
                st.error("Failed to process the image.")
        except Exception as e:
            st.error(f"Error processing image: {e}")

if __name__ == "__main__":
    main()
