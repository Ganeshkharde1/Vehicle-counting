import cv2

# Load video from file or camera
cap = cv2.VideoCapture('video.mp4')  # replace with your video file or camera index

# Initialize vehicle count and frame counter
vehicle_count = 0
frame_count = 0

# Create background subtractor object
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Loop over video frames
while True:
    # Read frame from video
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction to extract foreground objects
    fg_mask = bg_subtractor.apply(frame)

    # Apply thresholding to remove noiseq
    thresh = cv2.threshold(fg_mask, 128, 255, cv2.THRESH_BINARY)[1]

    # Find contours of foreground objects
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over contours and check if they are vehicles
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Check if contour is large enough to be a vehicle
        if w > 50 and h > 50:
            # Draw bounding box around vehicle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Increment vehicle count
            vehicle_count += 1

    # Increment frame count
    frame_count += 1
    print(f'Total vehicles: {vehicle_count}')
    # Show frame with bounding boxesq
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
    

# Release video and close window
cap.release()
cv2.destroyAllWindows()

# Print vehicle count and average vehicles per frame
print(f'Total vehicles: {vehicle_count}')
print(f'Average vehicles per frame: {vehicle_count / frame_count:.2f}')
