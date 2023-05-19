import tkinter as tk

def create_window():
    # Create the main window
    window = tk.Toplevel()
    window.geometry("500x500")
    root = tk()

    # Create a canvas widget
    canvas = tk.Canvas(window, width=500, height=500)

    # Load the image
    background_image = root.PhotoImage(file="E:/ASU6/Artificial Intelligence CSE472/ActionDetectionforSignLanguage-main/ActionDetectionforSignLanguage-main/back.png")

    # Add the image to the canvas
    canvas.create_image(0, 0, anchor=tk.NW, image=background_image)

    # Pack the canvas to fill the entire window
    canvas.pack(fill=tk.BOTH, expand=True)

    # Add other widgets or content to the window

    # Start the tkinter event loop
    window.mainloop()

def open_text_to_sign_window():
    text_to_sign_window = tk.Toplevel(root)
    text_to_sign_window.title("Text to Sign Window")
    # Add widgets and customize the text to sign window here

def open_sign_to_text_window():
    sign_to_text_window = tk.Toplevel(root)
    sign_to_text_window.title("Sign to Text Window")
    # Add widgets and customize the sign to text window here

root = tk.Tk()

# Set the window size
root.geometry("1500x1500")

# Create a frame to hold the buttons
frame = tk.Frame(root)
frame.pack(pady=200)  # Add vertical padding to center the frame

# Load the image
#root=tk()
background_image = root.PhotoImage(file="E:/ASU6/Artificial Intelligence CSE472/ActionDetectionforSignLanguage-main/ActionDetectionforSignLanguage-main/back.png")

# Create a canvas widget
canvas = tk.Canvas(root, width=1500, height=1500)
canvas.create_image(0, 0, anchor=tk.NW, image=background_image)
canvas.pack()

# Create two buttons and pack them in the frame
button1 = tk.Button(root, text="Text to Sign", width=50, height=10, font=("Arial", 20), command=open_text_to_sign_window)
button1.place(x=400, y=700)  # Set the button position using absolute coordinates

button2 = tk.Button(root, text="Sign to Text", width=50, height=10, font=("Arial", 20), command=open_sign_to_text_window)
button2.place(x=900, y=700)  # Set the button position using absolute coordinates

root.mainloop()
