# 🚗 Vehicle Detection & Classification

A deep learning-based vehicle detection and classification system built with TensorFlow/Keras and OpenCV. The model identifies and classifies vehicles in images into 8 categories, drawing bounding boxes with predicted labels.

## Features

- Classifies vehicles into 8 types: **City Car**, **Big Truck**, **Multi Purpose Vehicle**, **Sedan**, **Sports**, **Traffic**, **Truck**, and **Van**
- Draws bounding boxes with predicted labels on detected vehicles
- Normalizes and resizes input images for consistent model inference
- Saves annotated output images to disk

## Project Structure

```
pe01/
├── vehicle_detection.py      # Main inference script
├── firststeps.ipynb          # Jupyter notebook (model training / exploration)
├── vehicledetectmodel.h5     # Pre-trained Keras model
├── traffic.jpg               # Sample traffic scene image
├── car.jpg                   # Sample car image
├── cars.jpg                  # Sample cars image
├── truck.jpg                 # Sample truck image
├── van.jpg                   # Sample van image
├── vehicle.jpg               # Sample vehicle image
└── output.jpg                # Output image with detections
```

## Requirements

- Python 3.7+
- TensorFlow / Keras
- OpenCV (`cv2`)
- NumPy

Install dependencies:

```bash
pip install tensorflow opencv-python numpy
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/shiv2130/pe01.git
cd pe01
```

2. Update the `image_path` variable in `vehicle_detection.py` to point to your target image:

```python
image_path = 'path/to/your/image.jpg'
```

3. Define your bounding boxes (regions of interest) in the `bounding_boxes` list:

```python
bounding_boxes = [
    (x, y, width, height),  # one tuple per vehicle region
]
```

4. Run the script:

```bash
python vehicle_detection.py
```

The script will display the annotated image in a window and save the result as `output.jpg`.

## How It Works

1. An image is loaded using OpenCV.
2. For each bounding box defined, the corresponding region is cropped and resized to **48×48 pixels**.
3. Pixel values are normalized to the range `[0, 1]`.
4. The pre-trained Keras model (`vehicledetectmodel.h5`) predicts the vehicle class.
5. Bounding boxes and class labels are drawn on the original image using OpenCV.

## Model

The model is stored as `vehicledetectmodel.h5` (Keras HDF5 format). It was trained to classify 48×48 vehicle patches into one of 8 classes. See `firststeps.ipynb` for training details and experimentation.

## Sample Output

The script produces an annotated image similar to the included `output.jpg`, with green bounding boxes and class labels overlaid on detected vehicles.

## Notes

- Bounding boxes are currently defined manually. Integration with an object detector (e.g., YOLO or Faster R-CNN) would enable fully automatic detection.
- This project is part of a larger **Smart Adaptive Traffic Management System**.
