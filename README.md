# 🎭 Beautiful Face Detection App

> 🌟 A modern, user-friendly Python application for real-time face detection with cool visual effects!

## 📝 Description

This application provides an elegant GUI for detecting faces, eyes, and smiles in real-time using your webcam or video files. With a modern interface and multiple visual filters, it's perfect for both fun and educational purposes!

## ✨ Features

- 📹 Support for webcam and video file input
- 👁️ Real-time face, eye, and smile detection
- 🎨 Multiple visual filters:
  - Grayscale
  - Sepia
  - Blur
  - Edge Detection
- 📸 Take and save snapshots
- ⚙️ Adjustable detection parameters
- 🖌️ Modern and intuitive user interface

## 🚀 Installation

### Prerequisites

- ✅ Python 3.6+
- ✅ PyCharm or any Python IDE (recommended)

### Setup

1. 📦 Clone the repository:
   ```bash
   git clone https://github.com/yourusername/face-detection-app.git
   ```

2. 📂 Navigate to the project directory:
   ```bash
   cd face-detection-app
   ```

3. 🔧 Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install packages manually:
   ```bash
   pip install opencv-python pillow
   ```

## 🎮 Usage

1. 🚀 Run the application:
   ```bash
   python main.py
   ```

2. 🔍 Use the interface to:
   - Select camera or video file input
   - Toggle face detection on/off
   - Select visual filters
   - Enable eye and smile detection
   - Adjust detection parameters
   - Take snapshots

## 📊 Example

![Application Screenshot](screenshot.png)

## ⚡ Quick Controls

- 🎚️ **Scale Factor**: Adjust to detect faces at different scales
- 🧮 **Min Neighbors**: Adjust to fine-tune detection accuracy
- 🔍 **Checkboxes**: Enable/disable eye and smile detection
- 🎬 **Source**: Switch between webcam and video files
- 🖼️ **Filter**: Apply different visual effects
- 📸 **Snapshot**: Capture and save the current frame

## 🛠️ Technical Details

This application uses:
- 🐍 Python
- 🎯 OpenCV for face detection and image processing
- 🖼️ Tkinter for the graphical user interface
- 🔄 PIL for image manipulation

The face detection is powered by Haar Cascade classifiers, a machine learning object detection method used to identify objects in images or video.

## 📝 Contributing

1. 🍴 Fork the repository
2. 🔧 Create your feature branch (`git checkout -b feature/amazing-feature`)
3. 💾 Commit your changes (`git commit -m 'Add some amazing feature'`)
4. 📤 Push to the branch (`git push origin feature/amazing-feature`)
5. 🔁 Create a Pull Request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- 🧠 OpenCV community for the amazing computer vision library
- 🎨 Tkinter developers for the GUI toolkit
- 🚀 Everyone who has contributed to this project

---

Made with ❤️ by Your Name
