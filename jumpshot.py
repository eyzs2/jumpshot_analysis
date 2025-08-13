import mediapipe as mp
import numpy as np
import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions.drawing_utils import DrawingSpec

mp_pose = mp.solutions.pose                 # Pose detection module
mp_drawing = mp.solutions.drawing_utils     # Drawing utilities
mp_drawing_styles = mp.solutions.drawing_styles  # Default styles for drawing landmarks

# Helper function for angle calculation

def calculate_angle(landmarks):
    # Takes in a tuple of three coordinate objects, returns angle 
    # Numpy 2D array? cross product???
    # middle is apex of triangle
    xycoords = [np.array([landmark.x,landmark.y]) for landmark in landmarks]
    xyzcoords = [np.array([landmark.x,landmark.y,landmark.z]) for landmark in landmarks]

    # Computes the vectors from the coords, stores it in a list
    xyvectors = [(xycoords[0]-xycoords[1]),(xycoords[2]-xycoords[1])]
    xyzvectors = [(xyzcoords[0]-xyzcoords[1]),(xyzcoords[2]-xyzcoords[1])]
    
    xyangle = np.arccos(np.dot(xyvectors[0],xyvectors[1])/(np.linalg.norm(xyvectors[0])*np.linalg.norm(xyvectors[1])))
    xyzangle = np.arccos(np.dot(xyzvectors[0],xyzvectors[1])/(np.linalg.norm(xyzvectors[0])*np.linalg.norm(xyzvectors[1])))
    return (xyangle, xyzangle)


# Helper function for video annotation

def draw_landmarks_with_hidden_face(rgb_image, detection_result):
    annotated_image = np.copy(rgb_image)

    if not detection_result.pose_landmarks:
        return annotated_image
    
    pose_landmarks = detection_result.pose_landmarks[0]

    # Copy default drawing style and connections
    custom_style = mp_drawing_styles.get_default_pose_landmarks_style()
    custom_connections = list(mp_pose.POSE_CONNECTIONS)

    excluded_landmarks = [
        mp_pose.PoseLandmark.LEFT_EYE,
        mp_pose.PoseLandmark.RIGHT_EYE,
        mp_pose.PoseLandmark.LEFT_EYE_INNER,
        mp_pose.PoseLandmark.RIGHT_EYE_INNER,
        mp_pose.PoseLandmark.LEFT_EAR,
        mp_pose.PoseLandmark.RIGHT_EAR,
        mp_pose.PoseLandmark.LEFT_EYE_OUTER,
        mp_pose.PoseLandmark.RIGHT_EYE_OUTER,
        mp_pose.PoseLandmark.NOSE,
        mp_pose.PoseLandmark.MOUTH_LEFT,
        mp_pose.PoseLandmark.MOUTH_RIGHT,
        mp_pose.PoseLandmark.LEFT_PINKY,
        mp_pose.PoseLandmark.RIGHT_PINKY,
        mp_pose.PoseLandmark.LEFT_THUMB,
        mp_pose.PoseLandmark.RIGHT_THUMB
    ]

    # Make excluded landmarks invisible by setting thickness to None
    for landmark in excluded_landmarks:
        custom_style[landmark] = DrawingSpec(color=(255,255,0), thickness=None)

        # Remove any connections involving excluded landmarks
        custom_connections = [connection_tuple for connection_tuple in custom_connections 
                            if landmark.value not in connection_tuple]

    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      custom_connections,
      custom_style)

    return annotated_image

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Replace with path of video to be analysed

videoPath = "js_footage/klay_js.mov"

cap = cv2.VideoCapture(videoPath)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

video_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if not fps:
    print("Error: fps zero, exiting program")
    exit()

# Create a pose landmarker instance with the video mode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/pose_landmarker_full.task"),
    running_mode=VisionRunningMode.VIDEO)

# Output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# Replace path with desired video name
output_video_path = "js_footage/klay_js_annotated.mov"
out = cv2.VideoWriter(output_video_path, fourcc, int(fps/2), (video_width, video_height))

with PoseLandmarker.create_from_options(options) as landmarker:
    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame with timestamp in ms
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect_for_video(mp_image, frame_index * int(1000/fps))

        # Draw landmarks on RGB frame
        if result.pose_landmarks:
            annotated_rgb = draw_landmarks_with_hidden_face(rgb_frame, result)
        else:
            annotated_rgb = rgb_frame

        # Convert back to BGR for OpenCV
        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

        # Write annotated frame to output video
        out.write(annotated_bgr)

        frame_index += 1

# Cleanup
cap.release()
out.release()
print(f"Annotated video saved to {output_video_path}")