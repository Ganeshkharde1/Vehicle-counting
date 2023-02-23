import cv2
def loppp(vehicle_count):

    while True:
        # Read frame from video
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect vehicles in frame
        cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                            minSize=min_size, maxSize=max_size)

        # Draw bounding boxes around detected vehicles
        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Count number of vehicles
        vehicle_count += len(cars)

        # Display vehicle count and frame
        cv2.putText(frame, f'Total vehicles: {vehicle_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Vehicle Counting', frame)
        if(vehicle_count >= 40):
            print(0)
            vehicle_count=0
        # Wait for key press or exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

# Load video from file or camera
video_path = 'road.mp4'
cap = cv2.VideoCapture('video.mp4')

# Define vehicle detection parameters
car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')
min_size = (50, 50)
max_size = (200, 200)

# Initialize vehicle count
vehicle_count = 0
loppp(vehicle_count)
# Loop through video frames

# while(1):
#     a=input()
#     if(a== ord('q')):
#         print(3)
#         break
#     else:
#         loppp(vehicle_count)


# Release video and close windows
cap.release()
cv2.destroyAllWindows()
