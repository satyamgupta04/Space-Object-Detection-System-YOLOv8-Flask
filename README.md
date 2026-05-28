# 🚀 Space Object Detection System using YOLOv8 (Falcon Dataset)

🛰️ *Inspired by ISRO & NASA mission safety systems for real-time orbital situational awareness.*

---

## 📌 Overview

In space missions, **rapid identification of critical onboard objects** is essential for astronaut safety and emergency response.

This project presents a **real-time Space Object Detection System** powered by **YOLOv8 (Ultralytics)**, capable of detecting and classifying essential space-station equipment such as:

- 🔥 Fire Extinguishers  
- 🧪 Oxygen Tanks  
- 🧰 Toolboxes  
- ⚙️ Other critical mission equipment  

> 🎯 Goal: Enable fast, reliable, and automated object recognition to assist astronaut decision-making in emergency scenarios.

---

## 🧠 Core Idea

This system bridges the gap between:
- 🛰️ Orbital Safety Monitoring (NASA / ISRO concepts like Project NETRA)
- 🤖 Modern Computer Vision using Deep Learning (YOLOv8)

It demonstrates how **AI can enhance situational awareness in space environments**.

---

## ⚙️ Tech Stack

🧑‍💻 Language: Python  
🔥 Deep Learning Model: YOLOv8 (Ultralytics)  
🌐 Backend: Flask  
🎨 Frontend: HTML, CSS (Space-themed UI)  
📦 Libraries:
- OpenCV
- NumPy
- Torch
- Ultralytics YOLOv8

---

## 🧬 Dataset

📂 **Falcon Dataset (Duality AI Hackathon)**  
🔗 https://falcon.duality.ai/secure/documentation

The dataset includes annotated space-related objects designed for object detection tasks in controlled environments.

📌 Key Features:
- Bounding box annotations
- Multiple object categories
- Real-world inspired spacecraft environment data

---

## 🏗️ System Architecture
Input Image 🖼️
↓
Preprocessing 🧼
(Resize + Normalization)
↓
YOLOv8 Model 🧠
(Feature extraction + object detection)
↓
Bounding Box Prediction 📦
(Class label + confidence score)
↓
Post-processing 🎯
(NMS - Non Maximum Suppression)
↓
Flask Web App 🌐
(Display annotated output)

---

## 🔍 Model Details (YOLOv8)

- 📌 Architecture: One-stage object detector
- ⚡ Backbone: CSPDarknet-based feature extractor
- 🎯 Head: Decoupled detection head
- 📦 Loss Functions:
  - Box Loss (IoU-based)
  - Classification Loss
  - DFL (Distribution Focal Loss)

---

## 🚀 Features

✨ Real-time object detection  
✨ High accuracy under complex visual conditions  
✨ Lightweight inference using YOLOv8  
✨ Web-based image upload & visualization  
✨ Space-themed UI for immersive experience  
✨ Scalable for video stream detection  

---

## 🌌 Motivation

This project is inspired by:
- 🛰️ ISRO Project NETRA (Space Situational Awareness)
- 🚀 NASA spacecraft safety monitoring systems
- 🤖 AI-based autonomous mission assistance

It demonstrates how **computer vision can support astronaut safety and mission-critical operations**.

---

## 📊 Output Example

After uploading an image, the system returns:

- 📦 Detected object bounding boxes  
- 🏷️ Class labels (Fire Extinguisher, Oxygen Tank, Toolbox, etc.)  
- 🎯 Confidence scores  

---
##🔮 Future Improvements

-🚀 Integrate real-time video stream detection
-🛰️ Deploy on edge devices for spacecraft simulation
-📡 Extend dataset for orbital debris detection
-🧠 Replace YOLOv8 with YOLO-NAS / DETR for comparison
-☁️ Deploy on cloud (AWS / GCP) for scalable inference
-📱 Mobile-friendly detection interface

⸻

##🏆 Key Highlights

-Real-world inspired aerospace AI application
- Built using state-of-the-art YOLOv8
- Full-stack integration (Flask + Deep Learning)
- Strong relevance to ISRO/NASA safety systems
- Demonstrates production-level computer vision pipeline

⸻

##👨‍💻 Author

Satyam Gupta
🎓 B.Tech Final Year Student
💡 AI/ML | Computer Vision | Deep Learning | DSA

## ⚙️ How to Run

```bash
# Clone repository
git clone https://github.com/your-username/space-object-detection-yolov8.git

# Navigate project
cd space-object-detection-yolov8

# Install dependencies
pip install -r requirements.txt

# Run Flask app
python app.py

