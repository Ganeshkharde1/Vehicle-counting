import cv2

# Load video stream
cap = cv2.VideoCapture('path/to/video/file.mp4')

# Create background subtractor
subtractor = cv2.createBackgroundSubtractorMOG2()

# Initialize vehicle count
vehicle_count = 0

# Loop through video frames
while cap.isOpened():
    # Read frame from video stream
    ret, frame = cap.read()
    if not ret:
        break
    
    # Apply background subtraction
    fg_mask = subtractor.apply(frame)
    
    # Apply morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours of foreground objects
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Loop through contours
    for contour in contours:
        # Calculate contour area
        area = cv2.contourArea(contour)
        
        # Ignore small contours
        if area < 1000:
            continue
        
        # Draw contour on frame
        cv2.drawContours(frame, [contour], 0, (0, 255, 0), 2)
        
        # Increment vehicle count
        vehicle_count += 1
    
    # Display frame with contours
    cv2.imshow('frame', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) == ord('q'):
        break

# Release video stream and close all windows
cap.release()
cv2.destroyAllWindows()

# Print vehicle count
print(f'Total vehicles: {vehicle_count}')
