# Import all the required libraries
import cv2
import pickle
from ultralytics import YOLO
from utils import measure_distance, get_center_bbox

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)  # Load the YOLO model from the given path

    # This method selects two players closest to the court keypoints from the first frame of the #detections. It then filters the tracking data across all frames to include only these two #selected players.
    def choose_and_filter_players(self, court_keypoints, player_detections):
        # Take player detections from the first frame
        player_detections_first_frame = player_detections[0]

        # Choose two players closest to the court keypoints
        chosen_player = self.choose_players(court_keypoints, player_detections_first_frame)

        # Initialize a list to store filtered detections for each frame
        filtered_player_detections = []

        # Iterate over each frame's player detections
        for player_det in player_detections:
            # Keep only the bounding boxes of the chosen players
            filtered_player_dict = {
                track_id: bbox for track_id, bbox in player_det.items() if track_id in chosen_player
            }
            # Append the filtered dictionary for the current frame
            filtered_player_detections.append(filtered_player_dict)

        # Return the filtered detections for all frames
        return filtered_player_detections


# This method calculates the distances between the center of each player's bounding box and # all court keypoints. It returns the track IDs of the two players that are closest to the #court.

    def choose_players(self, court_keypoints, player_dict):
        distances = []  # List to hold distances between players and court keypoints

        # Iterate through all tracked players and their bounding boxes
        for track_id, bbox in player_dict.items():
            # Compute the center of the player's bounding box
            player_center = get_center_bbox(bbox)

            # Initialize minimum distance to a large number
            min_distance = float('inf')

            # Iterate over court keypoints in (x, y) pairs
            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i + 1])

                # Measure distance from player center to this court keypoint
                distance = measure_distance(player_center, court_keypoint)

                # Update the minimum distance if this is closer
                if distance < min_distance:
                    min_distance = distance

            # Store the player's track ID and its closest distance to the court
            distances.append((track_id, min_distance))

        # Sort players by their closest distance to the court (ascending)
        distances.sort(key=lambda x: x[1])

        # Select the two closest players based on distance
        chosen_players = [distances[0][0], distances[1][0]]

        return chosen_players


    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True)[0]  # Run object tracking on the input frame and get the first result
        class_names = results.names  # Get the mapping of class IDs to class names (e.g., 0: 'person', 1: 'car', etc.)
        player_dict = {}  # Initialize an empty dictionary to store detected players

        for box in results.boxes:  # Loop through each detected bounding box
            track_id = int(box.id.tolist()[0])  # Extract the unique tracking ID for this object
            result = box.xyxy.tolist()[0]  # Get the bounding box coordinates (x1, y1, x2, y2)
            class_ids = box.cls.tolist()[0]  # Get the class ID of the detected object
            det_class_names = class_names[class_ids]  # Map the class ID to a human-readable class name

            if det_class_names == "person":  # Only consider objects classified as "person"
                player_dict[track_id] = result  # Add the player's track ID and bounding box to the dictionary

        return player_dict  # Return the dictionary containing tracked players and their positions


    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
                return player_detections
        
        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections


# The purpose of the draw_bboxes function is to visually annotate each frame of a video by drawing:

#   1. Bounding boxes around detected players.
#   2. Player IDs (track IDs) above each box.

# This function takes in:

# 1. video_frames: a list of raw video frames (images),
# 2. player_detections: a list of dictionaries containing tracked player IDs and their bounding box coordinates for each frame,

# and outputs:

# output_video_frames: the same frames but now with boxes and labels drawn on them, so you can visually track and identify each player across frames.


    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []  # Initialize an empty list to store the annotated frames

        # Loop through each frame and its corresponding player detection dictionary
        for frame, player_dict in zip(video_frames, player_detections):

            # Loop through each player detected in the frame
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox  # Unpack the bounding box coordinates

                # Draw the player ID text just above the bounding box
                cv2.putText(
                    frame, 
                    f"Player ID: {track_id}", 
                    (int(x1), int(y1) - 10),  # Position text slightly above the top-left corner of the box
                    cv2.FONT_HERSHEY_SIMPLEX,  # Font style
                    0.9,  # Font scale
                    (0, 0, 255),  # Text color (red)
                    2  # Thickness of the text
                )

                # Draw the bounding box around the player
                cv2.rectangle(
                    frame, 
                    (int(x1), int(y1)),  # Top-left corner of the bounding box
                    (int(x2), int(y2)),  # Bottom-right corner of the bounding box
                    (0, 0, 255),  # Box color (red)
                    2  # Thickness of the box
                )

            output_video_frames.append(frame)  # Append the annotated frame to the output list

        return output_video_frames  # Return the list of frames with drawn bounding boxes and player IDs
