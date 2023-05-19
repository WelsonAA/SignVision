import tkinter as tk
from PIL import ImageTk, Image
import tested
import video
def create_window():
    # Create the main window
    window = tk.Toplevel()
    window.geometry("500x500")

    # Create a canvas widget
    canvas = tk.Canvas(window, width=500, height=500)

    # Load the image
    background_image = ImageTk.PhotoImage(Image.open("signlanguage2.png"))

    # Add the image to the canvas
    canvas.create_image(0, 0, anchor=tk.NW, image=background_image)

    # Pack the canvas to fill the entire window
    canvas.pack(fill=tk.BOTH, expand=True)

    # Add other widgets or content to the window

    # Start the tkinter event loop
    window.mainloop()

def open_text_to_sign_window(root):
    video.main(root)
    # Add widgets and customize the text to sign window here

def open_sign_to_text_window():
    tested.main()
    # Add widgets and customize the sign to text window here

root = tk.Tk()

# Set the window size
root.geometry("1400x1400")

# Load the image
background_image = ImageTk.PhotoImage(Image.open("signlanguage2.png"))

# Create a canvas widget
canvas = tk.Canvas(root, width=1400, height=1400)
canvas.create_image(0, 0, anchor=tk.NW, image=background_image)
canvas.pack()

# Create a frame for the buttons
frame = tk.Frame(root, bg="white", bd=1, highlightbackground="white", highlightthickness=1)
frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# Create two transparent-like buttons within the frame
button1 = tk.Button(frame, text="Text to Sign", width=30, height=5, font=("Arial", 20), command=open_text_to_sign_window(root=root), bd=0, highlightthickness=2, highlightbackground="white", bg='#FAEBD7')
button1.pack(pady=10)

button2 = tk.Button(frame, text="Sign to Text", width=30, height=5, font=("Arial", 20), command=open_sign_to_text_window, bd=0, highlightthickness=2, highlightbackground="white", bg='#FAEBD7')
button2.pack(pady=10)

root.mainloop()