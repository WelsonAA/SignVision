# importing libraries
import time

import cv2
import tkinter as tk
def submit_text(text_entry):
    entered_text = text_entry.get()
    return entered_text

def main(root):
    frame = tk.Frame(root)
    frame.pack(padx=10, pady=10)

    label = tk.Label(frame, text="Enter your text:")
    label.pack()

    text_entry = tk.Entry(frame)
    text_entry.pack(pady=5)

    submit_button = tk.Button(frame, text="Submit", command=submit_text(text_entry))
    submit_button.pack(pady=5)

    x=str(input())
    arr=x.split()

    print(arr)
    # Create a VideoCapture object and read from input file
    for i in range(len(arr)):
        cap = cv2.VideoCapture(f'videos/{arr[i]}.mp4')
    # Check if camera opened successfully

        if(cap.isOpened() == False):
            print("Error opening video file")

        # Read until video is completed
        while (cap.isOpened()):

            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                # Display the resulting frame
                cv2.imshow(f'{arr[i]}', frame)

                # Press Q on keyboard to exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            # Break the loop
            else:
                break

        # When everything done, release
        # the video capture object
        cap.release()

        # Closes all the frames
        cv2.destroyAllWindows()


