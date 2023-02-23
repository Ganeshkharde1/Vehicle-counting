import cv2

# Load video from file or camera
cap = cv2.VideoCapture('video1.mp4')  # or use 0 for camera

# Initialize vehicle count and frame count
vehicle_count = 0
frame_count = 0

# Create background subtractor object
fgbg = cv2.createBackgroundSubtractorMOG2()

# Loop through video frames
while True:
    # Read frame from video
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction to get foreground mask
    fgmask = fgbg.apply(frame)

    # Apply morphological operations to remove noise and fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the foreground mask
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through contours
    for contour in contours:
        # Compute bounding box and area of contour
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        # If contour is large enough to be a vehicle, increment count
        if area > 500:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            vehicle_count += 1

    # Increment frame count
    frame_count += 1

    # Display frame with bounding boxes and vehicle count
    cv2.putText(frame, f'Vehicles: {vehicle_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('frame', frame)

    # Break if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video and close window
cap.release()
cv2.destroyAllWindows()
