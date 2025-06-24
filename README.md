# 🚗 VahakAI – Driver Monitoring System

A real-time Driver Monitoring System (DMS) built using YOLOv8, capable of detecting **Open Eyes**, **Closed Eyes**, **Mobile Phone**, **Seatbelt**, and **Cigarette** usage. The system uses deep learning to predict driver states and displays results through a user-friendly GUI with **confidence scores**, along with options for **recording** and **snapshot saving**.

---

## 📌 Key Features

- 🔍 **Real-time Detection** of 5 critical driver behaviors.
- 🧠 Powered by **YOLOv8** for accurate and fast predictions.
- 🖥️ **Simple GUI** to visualize predictions live with bounding boxes and class scores.
- 💾 **Snapshot & Recording** options to store crucial events.
- 📈 Includes evaluation metrics like **confusion matrix** and **precision-recall curves**.
- ✅ Easily extensible for other driver-related classes or tasks.

---

## 🧠 Predicted Classes

- 👁️ Open Eyes  
- 😴 Closed Eyes  
- 📱 Using Mobile Phone  
- 🪖 Wearing Seatbelt  
- 🚬 Smoking Cigarette  

---

## 🖼️ Sample Output

### 🔍 Detection Screenshots

<p float="left">
  <img src="images/Screenshot 2025-05-28 231944.png" width="32%">
  <img src="images/Screenshot 2025-05-28 232009.png" width="32%">
  <img src="images/Screenshot 2025-05-28 232037.png" width="32%">
</p>

### 📊 Evaluation Results

<p float="left">
  <img src="images/bbox_image.png" width="32%">
  <img src="images/confusion_matrix.png" width="32%">
  <img src="images/precision_recall.png" width="32%">
</p>

---

## 🚀 Getting Started

Follow these steps to set up and run the project:

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/VahakAI.git
cd VahakAI
```

### 2. Install Required Dependencies

Make sure Python 3.8+ is installed. Then install dependencies using:
```bash

pip install opencv-python ultralytics matplotlib numpy
```

### 3. Add YOLOv8 Weights

Ensure that the yolov8n.pt file is present in the root directory. You can download it from Ultralytics if not already available.

### 4. ▶️ Running the Project

Launch the GUI and start real-time monitoring with:
```bash
python enhanced_dms.py
```
This script:

- Opens webcam feed

- Displays detection results with bounding boxes and confidence scores

- Supports user controls for snapshot and recording

### 💾 Snapshot & Recording Controls
| Key | Action                     |
| --- | -------------------------- |
| `R` | Start/Stop video recording |
| `S` | Save a snapshot            |

All files are timestamped and saved locally in the same directory.

### 🗂️ Project Structure
```plaintext
VahakAI/
├── YOLOv8/
│   └── yolov8n/
│       ├── args.yaml
│       ├── labels.jpg
│       └── labels_correlogram.jpg
├── images/
│   ├── Screenshot*.png
│   ├── bbox_image.png
│   ├── confusion_matrix.png
│   └── precision_recall.png
├── dms_yolo_model.pt
├── dms_yolo_predict_all.py
├── dms_yolo_project.py
├── enhanced_dms.py          <-- Main executable
├── yolov8n.pt               <-- YOLOv8 weights
└── README.md
```

📁 Note: Dataset and output_images/ folder are excluded from this repo due to size/privacy concerns.

### 🔮 Future Improvements
- 🚨 Add drowsiness alarm and voice alerts

- 📊 Log detections in CSV or database

- 🧩 Expand class labels (e.g., yawning, distraction)

- ☁️ Deploy on Raspberry Pi or edge devices

- 🌐 Add web-based UI with Streamlit or Flask
