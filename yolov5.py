import cv2
import torch
import argparse

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--video_path', type=str, default='road.mp4', help='path to input video')
parser.add_argument('--weights_path', type=str, default='yolov5s.pt', help='path to YOLOv5 weights file')
parser.add_argument('--confidence_threshold', type=float, default=0.5, help='confidence threshold for object detection')
parser.add_argument('--count_threshold', type=int, default=10, help='number of frames a vehicle must be detected to count')
args = parser.parse_args()

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).autoshape()

# Load video from file or camera
cap = cv2.VideoCapture(args.video_path)

# Initialize vehicle count and detected vehicles dictionary
vehicle_count = 0
detected_vehicles = {}

# Loop through video frames
while True:
    # Read frame from video
    ret, frame = cap.read()
    if not ret:
        break
        
    # Perform object detection on frame
    results = model(frame, conf=args.confidence_threshold)
    
    # Process detections and count vehicles
    for detection in results.xyxy[0]:
        label = int(detection[5])
        if label == 2:  # car label
            bbox = tuple(detection[:4].tolist())
            if bbox in detected_vehicles:
                detected_vehicles[bbox] += 1
                if detected_vehicles[bbox] >= args.count_threshold:
                    vehicle_count += 1
                    del detected_vehicles[bbox]
            else:
                detected_vehicles[bbox] = 1
    
    # Display vehicle count and frame
    cv2.putText(frame, f'Total vehicles: {vehicle_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Vehicle Counting', frame)
    
    # Wait for key press or exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release video and close windows
cap.release()
cv2.destroyAllWindows()
