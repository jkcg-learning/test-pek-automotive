import os
import cv2
import requests
from ultralytics import YOLO
from typing import List, Tuple

class AppleDetector:
    def __init__(self, model_path: str, model_url: str, input_dir: str, output_dir: str, apple_class_index: int = 47):
        self.model_path = model_path
        self.model_url = model_url
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.apple_class_index = apple_class_index

        # Ensure directories exist
        self._ensure_directory(self.output_dir)
        self._ensure_directory(os.path.dirname(self.model_path))
        
        # Download the model if not present
        self._download_model_if_needed()

        # Load the YOLO model
        self.model = YOLO(self.model_path)

    def _ensure_directory(self, directory: str):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def _download_model_if_needed(self):
        if not os.path.exists(self.model_path):
            print(f"Model not found at {self.model_path}. Downloading from {self.model_url}...")
            response = requests.get(self.model_url, stream=True)
            if response.status_code == 200:
                with open(self.model_path, 'wb') as model_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        model_file.write(chunk)
                print(f"Model downloaded and saved to {self.model_path}.")
            else:
                print(f"Failed to download model. HTTP status code: {response.status_code}")

    def detect_apples_in_image(self, image_path: str) -> Tuple[int, List[Tuple[int, int]], str]:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"Unable to read image {image_path}")
            return 0, [], ""

        results = self.model.predict(image_path)
        detections = results[0].boxes.data.cpu().numpy()

        apple_centers = []
        apple_count = 0

        for detection in detections:
            x1, y1, x2, y2, confidence, class_id = detection[:6]
            if int(class_id) == self.apple_class_index:
                apple_count += 1
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                apple_centers.append((center_x, center_y))
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                label = f'Apple: {apple_count}'
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box for apples
                cv2.putText(img_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.circle(img_bgr, (center_x, center_y), 5, (255, 0, 0), -1)  # Blue center point

        output_path = os.path.join(self.output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, img_bgr)

        return apple_count, apple_centers, output_path

    def process_images(self):
        image_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            print(f"No images found in directory {self.input_dir}")
            return

        for image_file in image_files:
            image_path = os.path.join(self.input_dir, image_file)
            apple_count, apple_centers, output_path = self.detect_apples_in_image(image_path)
            print(f"Image processed:  {image_path}")
            print(f"Number of apples detected: {apple_count}")
            print(f"Detected apple centers (x, y): {apple_centers}")
            print(f"Output saved to: {output_path}")

# Example usage
def main():
    model_filename = 'yolov8x.pt'
    model_path = os.path.join('models', model_filename)
    model_url = f'https://github.com/ultralytics/yolov8/releases/download/v1.0/{model_filename}'  # Example URL
    input_dir = '/content/input_images'  # Directory containing input images
    output_dir = '/content/output_images'  # Directory to save processed images

    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} does not exist")
        return

    apple_detector = AppleDetector(model_path, model_url, input_dir, output_dir)
    apple_detector.process_images()

if __name__ == "__main__":
    main()
