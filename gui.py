import time
from tkinter import *
from PIL import ImageTk, Image
import tested
import video

root = Tk()
root.title("SignVision")


def create_window():
    # Create the main window
    window = Tk.Toplevel()
    window.geometry("500x500")

    # Create a canvas widget
    canvas = Tk.Canvas(window, width=500, height=500)

    # Load the image
    background_image = ImageTk.PhotoImage(Image.open("signlanguage2.png"))

    # Add the image to the canvas
    canvas.create_image(0, 0, anchor=Tk.NW, image=background_image)

    # Pack the canvas to fill the entire window
    canvas.pack(fill=Tk.BOTH, expand=True)

    # Add other widgets or content to the window

    # Start the tkinter event loop
    window.mainloop()


def open_text_to_sign_window():
    video.main()





# Add widgets and customize the text to sign window here

def open_sign_to_text_window():
    tested.main()
    # Add widgets and customize the sign to text window here


# Set the window size
root.geometry("1400x1400")

# Load the image
background_image = ImageTk.PhotoImage(Image.open("signlanguage2.png"))

# Create a canvas widget
canvas = Canvas(root, width=1400, height=1400)

canvas.create_image(0, 0, anchor=NW, image=background_image)
canvas.pack()

# Create a frame for the buttons
frame = Frame(root, bg="white", bd=1, highlightbackground="white", highlightthickness=1)
frame.place(relx=0.5, rely=0.5, anchor=CENTER)

# Create two transparent-like buttons within the frame
button1 = Button(frame, text="Text to Sign", width=30, height=5, font=("Arial", 20), command=open_text_to_sign_window,
                 bd=0, highlightthickness=2, highlightbackground="white", bg='#FAEBD7')
button1.pack(pady=10)

button2 = Button(frame, text="Sign to Text", width=30, height=5, font=("Arial", 20), command=open_sign_to_text_window,
                 bd=0, highlightthickness=2, highlightbackground="white", bg='#FAEBD7')
button2.pack(pady=10)

root.mainloop()
