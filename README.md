# 1. Apple Detection App : Test PEK Automotive

This Streamlit app detects apples in images using YOLOv8 models. It displays the original image with bounding boxes around detected apples, along with the number of apples detected and their midpoint coordinates.

[![Watch the video]](https://github.com/jkcg-learning/test-pek-automotive/blob/main/extras/app_apple.mp4)

## Features
- Detect apples in images with different YOLO models.
- Display bounding boxes and midpoint coordinates for detected apples.
- Download YOLO models if not already available.
- Supports images in `.png`, `.jpg`, and `.jpeg` formats.

## Requirements
- Python 3.9.6
- Streamlit 1.35.0
- OpenCV 4.10.0.82
- Ultralyics YOLO 8.2.31

## How to Run

### Clone the Repository
```bash
git clone https://github.com/jkcg-learning/test-pek-automotive.git
```
### Navigate to the Project Directory
```bash
cd test-pek-automotive
```
### Install Dependencies
```bash
pip install -r requirements.txt
```
### Run the Streamlit App
```bash
streamlit run app.py
```
### Upload an Image
Use the file uploader in the Streamlit app interface to upload an image.

###  Select a Detection Model
Choose a YOLOv8 detection model from the dropdown menu.

### View Results
The app will display the original image with bounding boxes around detected apples, along with the number of apples detected and their midpoint coordinates.


# 2. Script mode

### Command Line Arguments

--input_dir: Directory containing the input images. Images should be in .png, .jpg, or .jpeg format.
--output_dir: Directory where the output images with detected apples will be saved.
--model_filename: Filename of the YOLO model to use for detection.

```bash
python detector.py --input_dir /path/to/input_images --output_dir /path/to/output_images --model_filename yolov8s.pt
```

## sample input 

![input_image](input_images/IMG_6214.JPG)

## sample output

![output_image](output_images/OUT_IMG_6214.jpg)
