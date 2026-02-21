#AYOYB
#ELABBADI
import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict

# Load YOLOv8 model
model = YOLO('yolov8s.pt')  # Fixed model name
names = model.names

# Open video
cap = cv2.VideoCapture("wrongside.mp4")

# COCO dataset vehicle classes
CAR_CLASS = 2  # Car
TRUCK_CLASS = 7  # Truck
BUS_CLASS = 5  # Bus
MOTORCYCLE_CLASS = 3  # Motorcycle (excluded)

# Track cars only (you can add other vehicle types if needed)
VEHICLE_CLASSES_TO_TRACK = [CAR_CLASS]  # Only cars
# VEHICLE_CLASSES_TO_TRACK = [CAR_CLASS, TRUCK_CLASS, BUS_CLASS]  # All large vehicles

# Tracking data
track_history = defaultdict(lambda: [])
car_count = 0
detected_cars = set()


# Debug mouse position
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse moved to: [{x}, {y}]")


cv2.namedWindow("Car Tracker")
cv2.setMouseCallback("Car Tracker", RGB)


def point_in_polygon(point, polygon):
    """Check if a point is inside a polygon"""
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def draw_info_panel(frame, current_cars, total_cars):
    """Draw information panel on the frame"""
    # Create semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (350, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Add text information
    cv2.putText(frame, "CAR TRACKING SYSTEM", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Current Cars: {current_cars}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(frame, f"Total Detected: {total_cars}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(frame, "Press any key for next frame, ESC to quit", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


frame_count = 0
area1 = [(297, 316), (288, 355), (526, 339), (518, 299)]
area2 = [(284, 364), (269, 404), (535, 389), (523, 346)]
total_detections = 0

print("Starting Car Tracking...")
print("Controls:")
print("- Any key: Next frame")
print("- ESC: Quit")
print("-" * 40)

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video")
        break

    frame_count += 1
    if frame_count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 600))

    # Track cars only
    results = model.track(frame, persist=True, classes=VEHICLE_CLASSES_TO_TRACK, verbose=False)

    current_cars = 0

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu().numpy()

        for track_id, box, class_id, conf in zip(ids, boxes, class_ids, confidences):
            if class_id in VEHICLE_CLASSES_TO_TRACK:
                x1, y1, x2, y2 = box

                # Calculate center point
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Store track history for trail
                track_history[track_id].append((cx, cy))
                if len(track_history[track_id]) > 20:  # Keep last 20 points
                    track_history[track_id].pop(0)

                # Count unique cars
                if track_id not in detected_cars:
                    detected_cars.add(track_id)
                    total_detections += 1
                    print(f"New car detected! ID: {track_id}")

                current_cars += 1

                # Get vehicle name
                vehicle_name = names[class_id]

                # Color based on confidence
                if conf > 0.7:
                    color = (0, 255, 0)  # Green for high confidence
                elif conf > 0.5:
                    color = (0, 255, 255)  # Yellow for medium confidence
                else:
                    color = (0, 0, 255)  # Red for low confidence

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw center point
                cv2.circle(frame, (cx, cy), 4, color, -1)

                # Draw label with track ID and confidence
                label = f"ID:{track_id} {vehicle_name} {conf:.2f}"
                cv2.putText(frame, label, (x1 + 3, y1 - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Draw tracking trail
                points = track_history[track_id]
                for i in range(1, len(points)):
                    cv2.line(frame, points[i - 1], points[i], color, 2)

                # Check if car is in specific areas
                in_area1 = point_in_polygon((cx, cy), area1)
                in_area2 = point_in_polygon((cx, cy), area2)

                if in_area1:
                    cv2.putText(frame, "AREA 1", (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                elif in_area2:
                    cv2.putText(frame, "AREA 2", (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Draw detection areas
    cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 0, 255), 2)
    cv2.polylines(frame, [np.array(area2, np.int32)], True, (0, 0, 255), 2)
    cv2.putText(frame, "Area 1", area1[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, "Area 2", area2[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Draw information panel
    draw_info_panel(frame, current_cars, total_detections)

    cv2.imshow("Car Tracker", frame)

    key = cv2.waitKey(0) & 0xFF
    if key == 27:  # ESC to quit
        break

print(f"\nTracking Summary:")
print(f"Total unique cars detected: {total_detections}")
print(f"Total frames processed: {frame_count // 3}")

cap.release()
cv2.destroyAllWindows()