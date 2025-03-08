import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import PIL.Image, PIL.ImageTk
import os
import numpy as np
import threading
from datetime import datetime


class FaceDetectionApp:
    def __init__(self, window, window_title):
        # Initialize main window
        self.window = window
        self.window.title(window_title)
        self.window.configure(bg="#f0f0f0")

        # Set window icon if available
        try:
            self.window.iconbitmap("app_icon.ico")
        except:
            pass

        # Set window size and position
        window_width = 1000
        window_height = 700
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        position_top = int(screen_height / 2 - window_height / 2)
        position_right = int(screen_width / 2 - window_width / 2)
        self.window.geometry(f"{window_width}x{window_height}+{position_right}+{position_top}")

        # Class variables
        self.video_source = 0  # Default camera
        self.detection_active = True
        self.filter_mode = "None"
        self.face_cascade = None
        self.eye_cascade = None
        self.smile_cascade = None
        self.load_cascades()

        # Create UI elements
        self.create_ui()

        # Open video source
        self.open_camera()

        # Start update loop
        self.update()

        # Initialize threading event for safe shutdown
        self.stopping = threading.Event()

        # Set the window close protocol
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def load_cascades(self):
        """Load the Haar cascade classifiers"""
        try:
            # Load face cascade
            face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
            if self.face_cascade.empty():
                raise Exception("Failed to load face cascade classifier")

            # Load eye cascade
            eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
            self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

            # Load smile cascade
            smile_cascade_path = cv2.data.haarcascades + 'haarcascade_smile.xml'
            self.smile_cascade = cv2.CascadeClassifier(smile_cascade_path)

        except Exception as e:
            messagebox.showerror("Error Loading Cascades", f"Error: {str(e)}")
            self.window.destroy()

    def create_ui(self):
        """Create the user interface"""
        # Create styles
        style = ttk.Style()
        style.configure("TButton", font=('Arial', 10))
        style.configure("TLabel", font=('Arial', 11), background="#f0f0f0")
        style.configure("Header.TLabel", font=('Arial', 14, 'bold'), background="#f0f0f0")

        # Main frame
        main_frame = ttk.Frame(self.window, padding=(10, 10, 10, 10))
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(main_frame, text="Advanced Face Detection", style="Header.TLabel")
        title_label.pack(pady=10)

        # Create frames for organization
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=5)

        video_frame = ttk.Frame(main_frame)
        video_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)

        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=5)

        # Source selection in top frame
        source_label = ttk.Label(top_frame, text="Video Source:")
        source_label.pack(side=tk.LEFT, padx=5)

        self.source_var = tk.StringVar(value="Camera")
        source_combo = ttk.Combobox(top_frame, textvariable=self.source_var,
                                    values=["Camera", "Video File"], width=15, state="readonly")
        source_combo.pack(side=tk.LEFT, padx=5)
        source_combo.bind("<<ComboboxSelected>>", self.change_source)

        browse_button = ttk.Button(top_frame, text="Browse...", command=self.browse_file)
        browse_button.pack(side=tk.LEFT, padx=5)

        # Video canvas in video frame
        self.canvas = tk.Canvas(video_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Controls in control frame
        # Detection toggle
        self.detection_var = tk.BooleanVar(value=True)
        detection_check = ttk.Checkbutton(control_frame, text="Face Detection",
                                          variable=self.detection_var,
                                          command=self.toggle_detection)
        detection_check.pack(side=tk.LEFT, padx=10)

        # Filter options
        filter_label = ttk.Label(control_frame, text="Filter:")
        filter_label.pack(side=tk.LEFT, padx=5)

        self.filter_var = tk.StringVar(value="None")
        filter_combo = ttk.Combobox(control_frame, textvariable=self.filter_var,
                                    values=["None", "Grayscale", "Sepia", "Blur", "Edge Detection"],
                                    width=15, state="readonly")
        filter_combo.pack(side=tk.LEFT, padx=5)
        filter_combo.bind("<<ComboboxSelected>>", self.change_filter)

        # Snapshot button
        snapshot_button = ttk.Button(control_frame, text="Take Snapshot", command=self.take_snapshot)
        snapshot_button.pack(side=tk.RIGHT, padx=10)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var,
                                 relief=tk.SUNKEN, anchor=tk.W)
        status_label.pack(fill=tk.X)

        # Face detection settings frame
        settings_frame = ttk.LabelFrame(main_frame, text="Detection Settings")
        settings_frame.pack(fill=tk.X, pady=10)

        # Scale Factor
        scale_label = ttk.Label(settings_frame, text="Scale Factor:")
        scale_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

        self.scale_var = tk.DoubleVar(value=1.1)
        scale_slider = ttk.Scale(settings_frame, from_=1.01, to=1.5,
                                 variable=self.scale_var, orient=tk.HORIZONTAL, length=200)
        scale_slider.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

        scale_value = ttk.Label(settings_frame, textvariable=tk.StringVar(value=lambda: f"{self.scale_var.get():.2f}"))
        scale_value.grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)

        # Min Neighbors
        neighbors_label = ttk.Label(settings_frame, text="Min Neighbors:")
        neighbors_label.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)

        self.neighbors_var = tk.IntVar(value=5)
        neighbors_slider = ttk.Scale(settings_frame, from_=1, to=10,
                                     variable=self.neighbors_var, orient=tk.HORIZONTAL, length=200)
        neighbors_slider.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

        neighbors_value = ttk.Label(settings_frame, textvariable=self.neighbors_var)
        neighbors_value.grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)

        # Also detect eyes and smile
        self.detect_eyes_var = tk.BooleanVar(value=False)
        eyes_check = ttk.Checkbutton(settings_frame, text="Detect Eyes",
                                     variable=self.detect_eyes_var)
        eyes_check.grid(row=0, column=3, padx=20, pady=5, sticky=tk.W)

        self.detect_smile_var = tk.BooleanVar(value=False)
        smile_check = ttk.Checkbutton(settings_frame, text="Detect Smile",
                                      variable=self.detect_smile_var)
        smile_check.grid(row=1, column=3, padx=20, pady=5, sticky=tk.W)

    def open_camera(self):
        """Open the video source"""
        try:
            # Release any existing video source
            if hasattr(self, 'vid') and self.vid is not None:
                self.vid.release()

            # Open new video source
            if self.source_var.get() == "Camera":
                self.vid = cv2.VideoCapture(self.video_source)
            else:
                if hasattr(self, 'video_file') and self.video_file:
                    self.vid = cv2.VideoCapture(self.video_file)
                else:
                    self.status_var.set("No video file selected")
                    return

            # Check if video source opened successfully
            if not self.vid.isOpened():
                raise Exception("Could not open video source")

            # Get video source dimensions
            self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

            self.status_var.set(f"Source: {'Camera' if self.source_var.get() == 'Camera' else self.video_file}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open video source: {str(e)}")
            self.status_var.set("Error: Could not open video source")

    def browse_file(self):
        """Open file browser to select video file"""
        filetypes = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*")
        ]

        filename = filedialog.askopenfilename(title="Select Video File",
                                              filetypes=filetypes)

        if filename:
            self.video_file = filename
            self.source_var.set("Video File")
            self.open_camera()

    def change_source(self, event=None):
        """Handle change of video source"""
        if self.source_var.get() == "Camera":
            self.video_source = 0
            self.open_camera()
        elif self.source_var.get() == "Video File":
            if not hasattr(self, 'video_file'):
                self.browse_file()
            else:
                self.open_camera()

    def change_filter(self, event=None):
        """Handle change of filter mode"""
        self.filter_mode = self.filter_var.get()
        self.status_var.set(f"Filter: {self.filter_mode}")

    def toggle_detection(self):
        """Toggle face detection on/off"""
        self.detection_active = self.detection_var.get()
        self.status_var.set(f"Face Detection: {'On' if self.detection_active else 'Off'}")

    def take_snapshot(self):
        """Save current frame as image file"""
        if hasattr(self, 'frame') and self.frame is not None:
            # Create directory if it doesn't exist
            if not os.path.exists('snapshots'):
                os.makedirs('snapshots')

            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snapshots/snapshot_{timestamp}.png"

            # Save the image
            cv2.imwrite(filename, self.frame)
            self.status_var.set(f"Snapshot saved: {filename}")

    def apply_filter(self, frame):
        """Apply selected filter to the frame"""
        if self.filter_mode == "None":
            return frame
        elif self.filter_mode == "Grayscale":
            return cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        elif self.filter_mode == "Sepia":
            kernel = np.array([[0.272, 0.534, 0.131],
                               [0.349, 0.686, 0.168],
                               [0.393, 0.769, 0.189]])
            return cv2.transform(frame, kernel)
        elif self.filter_mode == "Blur":
            return cv2.GaussianBlur(frame, (15, 15), 0)
        elif self.filter_mode == "Edge Detection":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return frame

    def detect_faces(self, frame):
        """Detect and highlight faces in the frame"""
        if not self.detection_active:
            return frame

        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Get current settings
        scale_factor = self.scale_var.get()
        min_neighbors = self.neighbors_var.get()

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Region of interest for face
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            # Detect eyes if enabled
            if self.detect_eyes_var.get():
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

            # Detect smile if enabled
            if self.detect_smile_var.get():
                smiles = self.smile_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.7,
                    minNeighbors=20,
                    minSize=(25, 25)
                )
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

            # Add face label
            cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Update status with face count
        self.status_var.set(f"Detected {len(faces)} faces")

        return frame

    def update(self):
        """Update the video frame"""
        if self.stopping.is_set():
            return

        if not hasattr(self, 'vid') or not self.vid.isOpened():
            self.status_var.set("No video source available")
            self.window.after(100, self.update)
            return

        # Get a frame from the video source
        ret, frame = self.vid.read()

        if ret:
            # Store the original frame
            self.frame = frame.copy()

            # Apply filter
            frame = self.apply_filter(frame)

            # Detect faces
            frame = self.detect_faces(frame)

            # Convert to format suitable for tkinter
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

            # Update canvas
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            # Clear previous image
            self.canvas.delete("all")

            # Calculate positioning to maintain aspect ratio
            img_ratio = frame.shape[1] / frame.shape[0]
            canvas_ratio = canvas_width / canvas_height

            if img_ratio > canvas_ratio:
                # Image is wider than canvas
                display_width = canvas_width
                display_height = canvas_width / img_ratio
            else:
                # Image is taller than canvas
                display_height = canvas_height
                display_width = canvas_height * img_ratio

            # Calculate position to center the image
            x_pos = (canvas_width - display_width) / 2
            y_pos = (canvas_height - display_height) / 2

            # Create image on canvas
            self.canvas.create_image(x_pos, y_pos, anchor=tk.NW, image=self.photo)
        else:
            # Video ended
            if self.source_var.get() == "Video File":
                self.status_var.set("Video ended")
                # Rewind video for replay
                self.vid.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Call this function again after 15 milliseconds
        self.window.after(15, self.update)

    def on_closing(self):
        """Handle window close event"""
        self.stopping.set()
        if hasattr(self, 'vid') and self.vid.isOpened():
            self.vid.release()
        self.window.destroy()


# Create and run the application
if __name__ == "__main__":
    # Create a tkinter window
    root = tk.Tk()

    # Create the application
    app = FaceDetectionApp(root, "Beautiful Face Detection App")

    # Start the GUI event loop
    root.mainloop()