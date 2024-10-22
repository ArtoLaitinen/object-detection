# YOLOv3 Object Detection using ONNX

This repository demonstrates object detection using the **YOLOv3** model in the ONNX format. The model is capable of detecting objects from the COCO dataset, including 80 different object classes.

- For more information about the YOLOv3 model in use, visit the official [ONNX GitHub page](https://github.com/onnx/models/tree/main/validated/vision/object_detection_segmentation/yolov3).
- You can also learn more about the ONNX (Open Neural Network Exchange) format on the official [ONNX website](https://onnx.ai/).

## Overview

This project uses the YOLOv3 object detection model, converted to the ONNX format, to detect objects in images. The model can identify objects such as people, vehicles, animals, and everyday objects based on the COCO dataset.

The inference code uses the **ONNX Runtime** for executing the model and **PIL (Python Imaging Library)** for image processing and drawing bounding boxes.

## Setup

To run this project locally, ensure that you have the following dependencies installed:

### Install Dependencies

1. Clone the repository:
   ```
   git clone https://github.com/ArtoLaitinen/object-detection.git
   cd onnx_yolov3_demo
   ```
2. Install the required Python packages:

   **Option A: Using `conda`:**

   If you prefer to use Anaconda (`conda`), follow these steps:
   1. Create and activate a conda environment (optional but recommended):
      ```
      conda create --name yolov3_env python=3.10
      conda activate yolov3_env
      ```
   2. **Install the required packages using `conda`:**
      ```
      conda install -c conda-forge onnxruntime pillow numpy
      ```

   **Option B: Using `pip`:**

   If you prefer to use `pip`, follow these steps:
      ```
      pip install onnxruntime pillow numpy
      ```
3. Download the YOLOv3 ONNX model from the [ONNX GitHub](https://github.com/onnx/models/tree/main/validated/vision/object_detection_segmentation/yolov3)
   - After downloading, place the model file in the project root directory (where demo.py is located)

## Running the Model

To run the YOLOv3 model on one of the provided sample images (e.g., `pedestrian.jpg`), execute the following Python script:

```
python demo.py
```

The script performs the following steps:

1. **Preprocessing:** The input image is resized with padding to match the model's input size (416x416) while maintaining the original aspect ratio.
2. **Model Inference:** The ONNX model is loaded using `onnxruntime`, and the preprocessed image is passed to the model for inference.
3. **Postprocessing:** The model's output is processed to obtain bounding boxes, class predictions, and confidence scores. These are scaled back to the original image size.
4. **Drawing:** Bounding boxes and class labels are drawn onto the original image and displayed.

## Files in the Repository

- **demo.py:** Main script for running the object detection model.
- **coco_classes.txt:** List of the 80 object classes the model can recognize (from the COCO dataset).
- **images:** (e.g., `pedestrian.jpg`) Sample images for testing the model.
- **.gitignore:** Git ignore file to exclude the large model file from the repository.

### Model

- The model `yolov3.onnx` can be downloaded from the [ONNX GitHub](https://github.com/onnx/models/tree/main/validated/vision/object_detection_segmentation/yolov3).

### COCO Classes

The list of object classes that the YOLOv3 model can detect is included in the file [coco_classes.txt](https://github.com/ArtoLaitinen/object-detection/blob/main/coco_classes.txt).

### Example Images

You can try running the model on the following sample images provided with the repository. The script will open the image with the detected objects and their bounding boxes drawn on it.

## Results

After running the script, the output image will display the objects detected by the YOLOv3 model with bounding boxes and class labels. The example below is generated using the `pedestrian.jpg` image from the sample images and shows the model detecting multiple objects, including a person, cars, and a handbag:

![Pedestrian](https://i.imgur.com/GCvqhnY.jpeg)
