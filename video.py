# importing libraries
import time

import cv2
from tkinter import*


def main():
    top = Toplevel()
    top.geometry("400x400")
    top.title("Text to Sign Live Translation")
    text = Text(top, width=40, height=10,)
    text.pack(side=LEFT)

    def translate_text():
        input_text = text.get("1.0", END)
        startTranslantion(input_text)

        # Call your translation function here\
    translateButton = Button(top, text="Translate", width=30, height=5, font=("Arial", 20),
                             command=translate_text)
    translateButton.pack(side=RIGHT)



def startTranslantion(x):
    print ("hi")
    print(x)
    str(x)
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


