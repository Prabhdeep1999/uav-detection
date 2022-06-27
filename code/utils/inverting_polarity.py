import cv2
import time

# video path to invert
video_path = "ENTER YOUR VIDEO PATH HERE"

# making a VideoCapture object
cap = cv2.VideoCapture(video_path)

# While loop utility
cap_opened = True

# We need to set resolutions so, convert them from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_size = (frame_width, frame_height)

# output video name
output_vid_name = video_path.split("/")[-1].split('.')[0] + "_inverse" + ".avi"

# making a VideoWriter object
vid_writer = cv2.VideoWriter(output_vid_name, cv2.VideoWriter_fourcc(*"MJPG"), 30, frame_size)

start_time = time.time()
# starting to loop over all the frames
while cap_opened:

    cap_opened, frame = cap.read()

    frame_inversed = cv2.bitwise_not(frame)

    cv2.imshow("orig", frame)
    cv2.imshow("inversed", frame_inversed)

    if time.time() - start_time > 10 and time.time() - start_time < 30:
        vid_writer.write(frame_inversed)
        # print(cap_opened)
        

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# releasing all the objects
cap.release()
vid_writer.release()
cv2.destroyAllWindows()