
"""
Project:- Movement Detection using Camera
Project Partners:- 1. Jagbir Singh Aulakh
                   2. Vraj SanjayKumar Patel
Course:- ENGI-9805 Computer Vision
"""

# Libraries Used


import pandas
from datetime import datetime
import imutils
import cv2
import numpy as np

# Initializing first_frame and make as static

first_frame = None

# Initializing status_list

status_list = [None, None]

# Initializing times that counts movements for data frame;
times = []

# define the maximum movement detection as persistence around 200
MOVEMENT_DETECTED_PERSISTENCE = 200

font = cv2.FONT_HERSHEY_DUPLEX
delay_counter = 0
movement_persistent_counter = 0

df = pandas.DataFrame(columns=["Start", "End"])

# Capturing the video using openCV library

video = cv2.VideoCapture(5)
video.release()
video = cv2.VideoCapture(0)

# Loop

while True:

    # Set transient motion detected as false
    transient_movement_flag = False

    # After the first frame is done the next frame is continued for capture
    check, frame = video.read()
    text = "Unoccupied"

    # If there's an error in capturing
    if not check:
        print("CAPTURE ERROR")
        continue

    # first frame status is = 0
    status = 0
    # Resizing the frame inorder to get a better view of all windows
    frame = imutils.resize(frame, width=700)

    # Implementation of grayscale method
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Applying a GaussianBlur inorder to remove noise in gray scale
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # In the while loop when first_frame is none
    # it would be assigned to gray to initialise it

    if first_frame is None:
        first_frame = gray
        continue

    delay_counter += 1

    # Otherwise, set the first frame to compare as the previous frame
    # But only if the counter reaches the appropriate value
    # The delay is to allow relatively slow motions to be counted as large
    # motions if they're spread out far enough

    if delay_counter > 10:
        delay_counter = 0
        first_frame = gray

    # Delta frame compares current frame(gray) with background/first frame
    # basically done the comparison in between frames

    delta_frame = cv2.absdiff(first_frame, gray)

    # Compare the two frames, find the difference using THRESH_BINARY method
    thresh_frame = cv2.threshold(delta_frame, 25, 255, cv2.THRESH_BINARY)[1]

    # Fill in holes via dilate() - smoothens the white areas
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    # RETR_EXTERNAL method is used to retrieve external of the contours in frame on the basis of hierarchy level.
    # CHAIN_APPROX_SIMPLE method is used for retrieving the contours in opencv2

    (cnts, _) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # loop over the contours

    for contour in cnts:

        # 10000 is for big objects being closer to camera
        # 10000 = 100x100 pixels window
        # can be set to a lower value like 2000 for micro movements

        if cv2.contourArea(contour) > 2000:
            transient_movement_flag = True
            continue

        # Status is assigned 1

        status = 1

        # Applying rectangle over found contours

        (x, y, w, h) = cv2.boundingRect(contour)

        # Draw a rectangle around big enough movements

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # The moment something moves momentarily, reset the persistent movement timer.

    if  transient_movement_flag == True:
        movement_persistent_flag = True
        movement_persistent_counter = MOVEMENT_DETECTED_PERSISTENCE

    # As long as there was a recent transient movement, say a movement was detected

    if movement_persistent_counter > 0:
        text = "Image is Moving [Detection occurred] " + str(movement_persistent_counter)
        movement_persistent_counter -= 1
    else:
        text = "Image is Still [No Detection]"

    # Print the text on the screen, and display the raw and processed video feeds

    cv2.putText(frame, str(text), (10, 35), font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    # Updating the status in status_list

    status_list.append(status)

    # To capture the last two items of status list that basically shows the difference between frames to detect object

    status_list = status_list[-2:]

    # we want to record datetime if status_list changes from 1 to 0

    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())

    # we want to record datetime if status_list changes from 0 to 1

    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())


    delta_frame = cv2.cvtColor(delta_frame, cv2.COLOR_GRAY2BGR)

    # Splice the two video frames together to make one long horizontal one

    cv2.imshow("Delta Frame & Color Frame", np.hstack((delta_frame, frame)))
    cv2.imshow("Gray Frame & Threshold Frame", np.hstack((gray, thresh_frame)))

    key = cv2.waitKey(1)

    if key == ord('q'):
        if status == 1:
            times.append(datetime.now())
        break

print(status_list)
print(times)

for i in range(0, len(times), 2):
    df = df.append({"Start": times[i], "End": times[i + 1]}, ignore_index=True)

df.to_csv("Times.csv")

# Exit all windows and cleanup memory
cv2.destroyAllWindows()
video.release()
