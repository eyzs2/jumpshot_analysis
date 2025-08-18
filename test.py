from jumpshot import calculate_angle, draw_landmarks_with_hidden_face
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2

# Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='models/pose_landmarker_full.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)



# Load the input image.
image = mp.Image.create_from_file("js_footage/js.jpg")

# Convert to OpenCV image
np_img = image.numpy_view()  # RGBA

bgr_img = cv2.cvtColor(np_img, cv2.COLOR_RGBA2BGR)
rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

# Detect pose landmarks from the input image.
detection_result = detector.detect(image)
pose_landmarks = detection_result.pose_landmarks[0]

annotated_img = draw_landmarks_with_hidden_face(rgb_img, detection_result)

# Convert to BGR for OpenCV display
annotated_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)

cv2.imshow("Annotated Image", annotated_bgr)
cv2.waitKey(0)  # Wait for key press
cv2.destroyAllWindows()

right_side = {"elbow": [12,14,16], "armpit": [24,12,14], "hip": [26,24,12], "knee": [28,26,24]}
left_side = {k: [v - 1 for v in vals] for k, vals in right_side.items()}


# If pose_landmarks[right_shoulder].z < pose_landmarks[left_shoulder].z
#   use right_arm_idx
# else use left

print(f"Right shoulder z: {pose_landmarks[12].z}, left shoulder z: {pose_landmarks[11].z}")

if pose_landmarks[12].z < pose_landmarks[11].z:
    coords = [pose_landmarks[idx] for idx in right_side["elbow"]]
else:
    coords = left_side["elbow"]
    coords = [pose_landmarks[idx] for idx in left_side["elbow"]]


angle = calculate_angle(coords)
deg_angle = np.rad2deg(angle)
print(f"Angle in radians: {angle}, angle in degrees: {deg_angle}")