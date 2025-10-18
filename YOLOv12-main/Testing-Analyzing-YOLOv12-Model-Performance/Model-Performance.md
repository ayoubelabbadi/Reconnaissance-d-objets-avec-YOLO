# YOLO Object Detection and Blurring Scripts (using Ultralytics)

## Description

This repository contains two Python scripts demonstrating real-time object detection using a YOLO model via the `ultralytics` library and OpenCV (`cv2`).

1.  **`main.py`**: Performs standard object detection on video files or webcam streams. It identifies objects, draws bounding boxes with class labels and confidence scores, calculates FPS, and saves the annotated video.
2.  **`obj_blurring.py`**: Extends the functionality of `main.py`. In addition to detection and annotation, it **blurs the area inside each detected bounding box** before drawing the box outline and label.

Both scripts process video frames, identify objects based on the COCO dataset, calculate and display the processing Frames Per Second (FPS), and save the resulting video.

**Note on "YOLOv12":** The scripts load a model named `yolo12n.pt`. As of this writing, "YOLOv12" is not an officially recognized version from the original YOLO authors or major research groups. This might be a custom-trained model, an unofficial variant, or potentially a naming convention specific to a certain project using the `ultralytics` framework (which commonly supports YOLOv5, YOLOv8, etc.). The underlying detection principles and usage with the `ultralytics` library remain consistent.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Standard Detection (`main.py`)](#standard-detection-mainpy)
  - [Detection with Blurring (`obj_blurring.py`)](#detection-with-blurring-obj_blurringpy)
- [Code Explanation Highlights](#code-explanation-highlights)
  - [Common Components (Both Scripts)](#common-components-both-scripts)
  - [Blurring Logic (`obj_blurring.py` only)](#blurring-logic-obj_blurringpy-only)
- [Configuration Parameters](#configuration-parameters)
- [Screenshots](#screenshots)
  - [Object Detection Example (`main.py`)](#object-detection-example-mainpy)
  - [Blurred Object Detection Example (`obj_blurring.py`)](#blurred-object-detection-example-obj_blurringpy)

## Features

**Both `main.py` and `obj_blurring.py`:**

*   Real-time object detection on video files or live webcam feed.
*   Uses a pre-trained YOLO model (`yolo12n.pt`) loaded via the `ultralytics` library.
*   Detects objects from the 80 classes in the COCO dataset.
*   Draws bounding boxes around detected objects.
*   Displays class labels and confidence scores for each detection.
*   Calculates and overlays the processing FPS on the video.
*   Saves the processed video with annotations to an output file (`output.mp4`).
*   Configurable confidence threshold and Non-Maximum Suppression (NMS) IoU threshold.

**`obj_blurring.py` Only:**

*   **Blurs the area inside detected bounding boxes** to anonymize or obscure objects.
*   Configurable blur intensity (`blur_ratio`).

## Prerequisites

*   Python 3.x
*   OpenCV library (`opencv-python`)
*   Ultralytics library (`ultralytics`)
*   A pre-trained YOLO model file compatible with Ultralytics (e.g., `yolo12n.pt` used in the scripts).
*   An input video file (e.g., `video.mp4`) or a connected webcam.

## Installation

1.  **Clone or download the repository/scripts.**
2.  **Install required Python libraries:**
    ```bash
    pip install opencv-python ultralytics
    ```
3.  **Obtain the model file:** Make sure you have the `yolo12n.pt` model file (or your desired model) in the same directory as the scripts, or provide the correct path within the scripts.
4.  **Prepare input video:** Place your input video file (e.g., `video.mp4`) in a `Resources/Videos/` subdirectory relative to the scripts, or modify the path in the `cv2.VideoCapture()` line within the desired script. Create the directories if they don't exist.

## Usage

### Standard Detection (`main.py`)

1.  **Configure `main.py` (Optional):**
    *   Modify `cv2.VideoCapture(...)` for your video source.
    *   Modify `cv2.VideoWriter(...)` for the output filename.
    *   Adjust `conf` and `iou` parameters in `model.predict(...)` if needed.
2.  **Run the script:**
    ```bash
    python main.py
    ```
3.  **Viewing:** An OpenCV window titled "Video" will show the stream with detected objects, bounding boxes, labels, and FPS.
4.  **Stopping:** Press '1' in the OpenCV window.
5.  **Output:** The processed video (without blurring) is saved as `output.mp4`.

### Detection with Blurring (`obj_blurring.py`)

1.  **Configure `obj_blurring.py` (Optional):**
    *   Modify `cv2.VideoCapture(...)` for your video source.
    *   Modify `cv2.VideoWriter(...)` for the output filename.
    *   Adjust `conf` and `iou` parameters in `model.predict(...)` if needed.
    *   Adjust the `blur_ratio` variable to control blur intensity (higher means more blur).
2.  **Run the script:**
    ```bash
    python obj_blurring.py
    ```
3.  **Viewing:** An OpenCV window titled "Video" will show the stream with detected objects **blurred inside their boxes**, along with box outlines, labels, and FPS.
4.  **Stopping:** Press '1' in the OpenCV window.
5.  **Output:** The processed video (with blurred objects) is saved as `output.mp4`.

## Code Explanation Highlights

### Common Components (Both Scripts)

*   **Imports**: `cv2`, `math`, `time`, `ultralytics.YOLO`.
*   **Video I/O**: `cv2.VideoCapture` to read, `cv2.VideoWriter` to save.
*   **Model Loading**: `model = YOLO("yolo12n.pt")`.
*   **Class Names**: `cocoClassNames` list for mapping IDs.
*   **Detection**: `results = model.predict(frame, conf=..., iou=...)`.
*   **Box/Label Drawing**: Extracting `box.xyxy`, `box.cls`, `box.conf`; using `cv2.rectangle` and `cv2.putText` to draw annotations.
*   **FPS Calculation**: Using `time.time()` difference between frames.
*   **Main Loop**: `while True` loop reading frames, processing, displaying, and writing.
*   **Cleanup**: `cap.release()`, `output_video.release()`, `cv2.destroyAllWindows()`.

### Blurring Logic (`obj_blurring.py` only)

Located inside the loop iterating through detected `boxes`:

```python
# --- Blurring Start (obj_blurring.py only) ---
x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # Ensure coordinates are int

# Extract the region of interest (ROI)
blur = frame[y1:y2, x1:x2]

# Apply blur to the ROI
# Check if ROI is valid before blurring
if blur.size > 0:
    blur_obj = cv2.blur(blur, (blur_ratio, blur_ratio))
    # Place the blurred ROI back into the main frame
    frame[y1:y2, x1:x2] = blur_obj
# --- Blurring End ---

# Draw rectangle *after* blurring
cv2.rectangle(frame, (x1, y1), (x2, y2), [255,0,0], 2)
# ... rest of label drawing ...
```
*   The key steps are slicing the frame to get the detected object's region (`frame[y1:y2, x1:x2]`), applying `cv2.blur` to that slice, and then putting the blurred slice back into the original frame.
*   A check `if blur.size > 0:` is added as good practice to avoid errors if the coordinates somehow result in an empty slice.

## Configuration Parameters

Modify these within the respective script (`main.py` or `obj_blurring.py`):

*   `cap = cv2.VideoCapture(...)`: Input video source.
*   `output_video = cv2.VideoWriter(...)`: Output video file configuration.
*   `model = YOLO(...)`: Path to the YOLO model file.
*   `conf=...` (in `model.predict`): Confidence threshold (0.0 to 1.0).
*   `iou=...` (in `model.predict`): NMS IoU threshold (0.0 to 1.0).
*   `blur_ratio = 50` (**`obj_blurring.py` only**): Kernel size for blurring intensity.


## Screenshots


### Object Detection Example (`main.py`)

#### Object Detection on Image Example

![Object Detection on Image Placeholder](../shared/Testing-And-Analyzing/image-boundaries.png)
*Caption: Example of object detection results on a single image frame.*

#### Object Detection on Video Example

![Object Detection on Video Placeholder](../shared/Testing-And-Analyzing/ezgif-273383aa0a9c94.gif)
*Caption: Example frame from the processed output video showing detected objects, labels, and FPS.*

### Blurred Object Detection Example (`obj_blurring.py`)

![Blurred Object Detection GIF Placeholder](../shared/Testing-And-Analyzing/obj_blurring.gif)
*Caption: Example GIF output from `obj_blurring.py` showing detected objects blurred within their bounding boxes.*