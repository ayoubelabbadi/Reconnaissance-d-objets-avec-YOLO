import cv2
import math
import time
from ultralytics import YOLO


def detect_objects_yolov12(video_path, model_path="models/yolov12m.pt", output_path="output/output.mp4", conf_thresh=0.15,
                           iou_thresh=0.1):
    # Load YOLOv12 model from the given model path
    model = YOLO(model_path)

    # Open the input video file
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))  # Get width of the video frames
    frame_height = int(cap.get(4))  # Get height of the video frames
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get frames per second of the video

    # Set up video writer to save the output with bounding boxes
    output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # COCO class names to map class IDs to human-readable labels
    cocoClassNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                      "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                      "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                      "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                      "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                      "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                      "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                      "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                      "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                      "teddy bear", "hair drier", "toothbrush"]

    ptime = 0  # Previous time used to calculate FPS
    count = 0  # Frame counter

    while True:
        ret, frame = cap.read()  # Read a single frame from the video
        if not ret:
            break  # Exit loop if no frame is returned (end of video)

        count += 1
        print(f"Frame Number: {count}")  # Print the current frame number for logging/debugging

        # Run object detection on the current frame using YOLOv12
        results = model.predict(frame, conf=conf_thresh, iou=iou_thresh)

        for result in results:  # Loop through detection results
            boxes = result.boxes  # Get all bounding boxes in this result
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert bounding box coordinates to integers
                cv2.rectangle(frame, (x1, y1), (x2, y2), [255, 0, 0], 2)  # Draw rectangle around detected object

                class_id = int(box.cls[0])  # Get class ID of detected object
                classname = cocoClassNames[class_id]  # Map class ID to class name using COCO list
                conf = round(float(box.conf[0]), 2)  # Get confidence score and round to 2 decimal places
                label = f"{classname}: {conf}"  # Create label text for the detection

                # Calculate size and position for label box

                # Get the size (width and height) of the text label using cv2.getTextSize
                # Arguments:
                #   label: the string to be drawn (e.g., "person: 0.89")
                #   fontFace: 0 is shorthand for cv2.FONT_HERSHEY_SIMPLEX
                #   fontScale: 0.5 scales the font size to half
                #   thickness: 2 is the line thickness of the font
                # getTextSize returns a tuple: ((text_width, text_height), baseline), so we extract [0] for size
                text_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]

                # Calculate the bottom-right corner of the label background rectangle
                #   x1 is the top-left x coordinate of the bounding box
                #   text_size[0] is the width of the label text
                #   y1 is the top-left y of the bounding box
                #   text_size[1] is the height of the label text
                #   Subtracting a few pixels for spacing (3 pixels)
                c2 = x1 + text_size[0], y1 - text_size[1] - 3

                # Draw a filled rectangle (background) behind the text label
                # Arguments:
                #   frame: the image to draw on
                #   (x1, y1): top-left corner of the rectangle (aligned with the bounding box)
                #   c2: bottom-right corner based on text dimensions
                #   [255, 0, 0]: rectangle color in BGR (blue)
                #   -1: thickness=-1 means fill the rectangle
                cv2.rectangle(frame, (x1, y1), c2, [255, 0, 0], -1)

                # Overlay the text label on top of the filled rectangle
                # Arguments:
                #   frame: the image to draw on
                #   label: the actual label string to draw (e.g., "dog: 0.76")
                #   (x1, y1 - 2): position of text baseline, slightly above the top-left corner
                #   0: font type (cv2.FONT_HERSHEY_SIMPLEX)
                #   0.5: font scale (same as before)
                #   [255, 255, 255]: font color in BGR (white)
                #   thickness=1: thin line thickness for readability
                #   lineType=cv2.LINE_AA: anti-aliased lines for smoother text
                cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)




        # Calculate FPS and show it on the frame

        # This timestamp marks when this frame is being processed
        ctime = time.time()# Get the current time in seconds 

        # Calculate the time difference between the current and previous frame
        # Then compute FPS as the reciprocal of that time difference (i.e., frames per second)
        fps = 1 / (ctime - ptime) if (ctime - ptime) > 0 else 0  # Avoid division by zero

        # Calculate the time difference between the current and previous frame
        # Then compute FPS as the reciprocal of that time difference (i.e., frames per second)
        ptime = ctime
        cv2.putText(frame, f"FPS: {int(fps)}", (30, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        # Save the processed frame to the output video
        output_video.write(frame)
        cv2.imshow("Video", frame)  # Show the frame in a window

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit if 'q' is pressed
            break

    cap.release()  # Release video capture
    output_video.release()  # Release video writer
    cv2.destroyAllWindows()  # Close all OpenCV windows

# Run the function on a sample video file
detect_objects_yolov12("Resources/Videos/video2.mp4")

