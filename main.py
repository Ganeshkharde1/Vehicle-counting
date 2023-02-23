import cv2
import numpy as np

# Load the video file
cap = cv2.VideoCapture('video.mp4')

# Create a Background Subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# Define a Region of Interest (ROI) for tracking vehicles
# In this example, we're tracking vehicles on the road
roi = [(0, 400), (800, 400), (800, 600), (0, 600)]

# Initialize variables to count vehicles
count = 0
is_vehicle_detected = [0, 0, 0, 0, 0, 0, 0, 0]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Crop the frame to the ROI
    cropped = frame[400:600, 0:800]

    # Apply background subtraction
    fgmask = fgbg.apply(cropped)

    # Apply morphological transformations to remove noise
    kernel = np.ones((3,3),np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    # Find contours in the foreground mask
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        # Check if the contour area is big enough to be a vehicle
        if cv2.contourArea(contour) > 500:

            # Get the bounding rectangle of the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Check if the vehicle is within the ROI
            if y > roi[0][1] and y+h < roi[2][1] and x > roi[0][0] and x+w < roi[1][0]:
                
                # Check if the vehicle has already been detected
                if not is_vehicle_detected[i]:
                    is_vehicle_detected[i] = 1
                    count += 1
                    print("Vehicle detected: ", count)

    # Reset the vehicle detection status after some frames
    if len(contours) == 0:
        is_vehicle_detected = [0, 0, 0, 0, 0, 0, 0, 0]

    # Display the video and mask frames
    cv2.imshow('Video', cropped)
    cv2.imshow('Mask', fgmask)
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
