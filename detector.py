import os
import cv2
import requests
import argparse
from ultralytics import YOLO
import json

class AppleDetector:
    """Class to detect apples in images using YOLO object detection."""

    def __init__(self, model_path: str, model_url: str, input_dir: str, output_dir: str, apple_class_index: int = 47):
        """
        Initialize the AppleDetector.

        Args:
            model_path (str): Path to the YOLO model file.
            model_url (str): URL to download the YOLO model if not found locally.
            input_dir (str): Directory containing input images.
            output_dir (str): Directory to save processed images.
            apple_class_index (int, optional): Index for the apple class in the YOLO model. Defaults to 47.
        """
        self.model_path = model_path
        self.model_url = model_url
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.apple_class_index = apple_class_index

        try:
            # Ensure directories exist
            self._ensure_directory(self.output_dir)
            self._ensure_directory(os.path.dirname(self.model_path))
            
            # Download the model if not present
            self._download_model_if_needed()

            # Load the YOLO model
            self.model = YOLO(self.model_path)
        except Exception as e:
            print(f"Error initializing AppleDetector: {e}")

    def _ensure_directory(self, directory: str):
        """Ensure that the given directory exists. If not, create it."""
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except Exception as e:
            print(f"Error ensuring directory {directory}: {e}")

    def _download_model_if_needed(self):
        """Download the YOLO model if not found locally."""
        if not os.path.exists(self.model_path):
            try:
                print(f"Model not found at {self.model_path}. Downloading from {self.model_url}...")
                response = requests.get(self.model_url, stream=True)
                if response.status_code == 200:
                    with open(self.model_path, 'wb') as model_file:
                        for chunk in response.iter_content(chunk_size=8192):
                            model_file.write(chunk)
                    print(f"Model downloaded and saved to {self.model_path}.")
                else:
                    print(f"Failed to download model. HTTP status code: {response.status_code}")
            except requests.RequestException as e:
                print(f"Error downloading the model: {e}")
            except IOError as e:
                print(f"Error saving the model: {e}")

    def detect_apples_in_image(self, image_path: str) -> tuple:
        """
        Detect apples in a single image.

        Args:
            image_path (str): Path to the input image.

        Returns:
            tuple: A tuple containing the number of apples detected, list of apple center coordinates, and the output image path.
        """
        try:
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                print(f"Unable to read image {image_path}")
                return 0, [], ""

            results = self.model.predict(image_path)
            detections = results[0].boxes.data.cpu().numpy()

            apple_centers = []
            apple_count = 0

            for detection in detections:
                try:
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
                except Exception as e:
                    print(f"Error processing detection {detection}: {e}")

            output_path = os.path.join(self.output_dir, os.path.basename(image_path))
            try:
                cv2.imwrite(output_path, img_bgr)
            except IOError as e:
                print(f"Error saving the output image: {e}")

            return apple_count, apple_centers, output_path
        except Exception as e:
            print(f"Error detecting apples in image {image_path}: {e}")
            return 0, [], ""

    def process_images(self):
        """
        Process all images in the input directory.

        This function detects apples in each image, saves the output images, and creates a JSON file with the detection results.
        """
        try:
            image_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not image_files:
                print(f"No images found in directory {self.input_dir}")
                return

            output_data = []
            for image_file in image_files:
                try:
                    image_path = os.path.join(self.input_dir, image_file)
                    apple_count, apple_centers, output_path = self.detect_apples_in_image(image_path)
                    output_data.append({
                        'image_name': image_file,
                        'apple_count': apple_count,
                        'apple_centers': apple_centers,
                        'output_path': output_path
                    })
                    print(f"Image processed: {image_path}")
                    print(f"Number of apples detected: {apple_count}")
                    print(f"Detected apple centers (x, y): {apple_centers}")
                    print(f"Output saved to: {output_path}")
                except Exception as e:
                    print(f"Error processing image {image_file}: {e}")
            
            output_json_path = os.path.join(self.output_dir, 'output_data.json')
            with open(output_json_path, 'w') as json_file:
                json.dump(output_data, json_file, indent=4)
            print(f"Output data saved to: {output_json_path}")
        except Exception as e:
            print(f"Error processing images: {e}")

def main():
    """
    Main function to parse command line arguments and run the AppleDetector.
    """
    parser = argparse.ArgumentParser(description="Detect apples in images.")
    parser.add_argument('--model_filename', type=str, default='yolov8x.pt', help='Filename of the YOLO model.')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input images.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save processed images.')

    args = parser.parse_args()

    model_path = os.path.join('models', args.model_filename)
   
    model_url = f'https://github.com/ultralytics/yolov8/releases/download/v1.0/{args.model_filename}'

    if not os.path.exists(args.input_dir):
        print(f"Input directory {args.input_dir} does not exist")
        return

    try:
        apple_detector = AppleDetector(model_path, model_url, args.input_dir, args.output_dir)
        apple_detector.process_images()
    except Exception as e:
        print(f"Error running the main function: {e}")

if __name__ == "__main__":
    main()
