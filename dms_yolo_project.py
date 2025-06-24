#!/usr/bin/env python
# coding: utf-8

# In[35]:


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
import streamlit as st
from PIL import Image
import yaml
import torch
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
import pandas as pd
import shutil

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Configuration
DATASET_ROOT = r"C:\Users\anshg\Documents\final project"
DATA_YAML = os.path.join(DATASET_ROOT, "data.yaml")
MODEL_PATH = os.path.join(DATASET_ROOT, "dms_yolo_model.pt")
IMG_SIZE = 640
EPOCHS = 10
BATCH_SIZE = 16
CLASSES = ["Open Eye", "Closed Eye", "Cigarette", "Phone", "Seatbelt"]
NUM_CLASSES = len(CLASSES)
RUNS_DIR = os.path.join(DATASET_ROOT, "runs")  # Directory for training logs/checkpoints

# Validate Dataset
def validate_dataset():
    """Validate dataset structure and image-label pairs."""
    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(DATASET_ROOT, split, "images")
        label_dir = os.path.join(DATASET_ROOT, split, "labels")
        
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Directory not found: {img_dir}")
        if not os.path.exists(label_dir):
            raise FileNotFoundError(f"Directory not found: {label_dir}")
        
        img_files = set(f.split('.')[0] for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png')))
        label_files = set(f.split('.')[0] for f in os.listdir(label_dir) if f.endswith('.txt'))
        
        missing_labels = img_files - label_files
        missing_images = label_files - img_files
        
        if missing_labels:
            print(f"Warning: {split}/images missing labels for: {len(missing_labels)} files")
        if missing_images:
            print(f"Warning: {split}/labels missing images for: {len(missing_images)} files")

# Data Preparation
def prepare_data_yaml():
    """Create data.yaml with absolute paths."""
    data_config = {
        'train': os.path.join(DATASET_ROOT, "train", "images"),
        'val': os.path.join(DATASET_ROOT, "valid", "images"),
        'test': os.path.join(DATASET_ROOT, "test", "images"),
        'nc': NUM_CLASSES,
        'names': CLASSES
    }
    with open(DATA_YAML, 'w') as f:
        yaml.dump(data_config, f)
    return DATA_YAML

# Model Training
def train_model(resume=False):
    """Train YOLOv8 model or load existing model."""
    # Check if trained model exists
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        return model, None
    
    device = 0 if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: Training on CPU. Consider enabling GPU for faster training.")
    
    # Check for previous training checkpoint to resume
    checkpoint_path = os.path.join(RUNS_DIR, "detect", "dms_yolo", "weights", "last.pt")
    if resume and os.path.exists(checkpoint_path):
        print(f"Resuming training from {checkpoint_path}")
        model = YOLO(checkpoint_path)
    else:
        # Clean up old training runs if starting fresh
        if os.path.exists(RUNS_DIR):
            print(f"Cleaning up old training runs at {RUNS_DIR}")
            shutil.rmtree(RUNS_DIR)
        print("Starting new training session with yolov8n.pt")
        model = YOLO("yolov8n.pt")
    
    # Train the model
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        name="dms_yolo",
        device=device,
        workers=4  # Add dataloader workers for faster data loading
    )
    model.save(MODEL_PATH)
    return model, results



# In[31]:


def plot_bounding_boxes(model, img_path, output_path="bbox_image.png"):
    """Visualize bounding boxes on a sample image."""
    img = cv2.imread(img_path)
    results = model.predict(img_path, conf=0.5)
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        
        for box, cls, score in zip(boxes, classes, scores):
            x1, y1, x2, y2 = map(int, box)
            label = f"{CLASSES[int(cls)]} {score:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imwrite(output_path, img)
    return img

def plot_precision_recall(model, test_dir):
    """Plot precision-recall curves for test set."""
    y_true, y_score = [], []
    for img_file in os.listdir(os.path.join(test_dir, "images")):
        img_path = os.path.join(test_dir, "images", img_file)
        label_file = os.path.join(test_dir, "labels", img_file.replace(".jpg", ".txt").replace(".png", ".txt"))
        
        true_classes = []
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    parts = line.split()
                    if len(parts) != 5:  # Expected format: <class_id> <x_center> <y_center> <width> <height>
                        print(f"Warning: Skipping malformed line in {label_file}: {line}")
                        continue
                    try:
                        class_id = int(parts[0])
                        if 0 <= class_id < NUM_CLASSES:
                            true_classes.append(class_id)
                        else:
                            print(f"Warning: Invalid class_id {class_id} in {label_file}, skipping")
                    except ValueError:
                        print(f"Warning: Invalid class_id (not an integer) in {label_file}: {line}, skipping")
        
        results = model.predict(img_path, conf=0.5)
        pred_classes = [int(cls) for result in results for cls in result.boxes.cls.cpu().numpy()]
        pred_scores = [score for result in results for score in result.boxes.conf.cpu().numpy()]
        
        true_bin = np.zeros(NUM_CLASSES)
        for cls in true_classes:
            true_bin[cls] = 1
        y_true.append(true_bin)
        
        score_bin = np.zeros(NUM_CLASSES)
        for cls, score in zip(pred_classes, pred_scores):
            score_bin[cls] = max(score_bin[cls], score)
        y_score.append(score_bin)
    
    y_true, y_score = np.array(y_true), np.array(y_score)
    
    plt.figure(figsize=(10, 8))
    for i in range(NUM_CLASSES):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_score[:, i])
        ap = average_precision_score(y_true[:, i], y_score[:, i])
        plt.plot(recall, precision, label=f"{CLASSES[i]} (AP={ap:.2f})")
    
    plt.title("Precision-Recall Curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.savefig(os.path.join(DATASET_ROOT, "precision_recall.png"))
    plt.close()

def plot_confusion_matrix(model, test_dir):
    """Plot confusion matrix for test set."""
    y_true, y_pred = [], []
    for img_file in os.listdir(os.path.join(test_dir, "images")):
        img_path = os.path.join(test_dir, "images", img_file)
        label_file = os.path.join(test_dir, "labels", img_file.replace(".jpg", ".txt").replace(".png", ".txt"))
        
        true_classes = []
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        print(f"Warning: Skipping malformed line in {label_file}: {line}")
                        continue
                    try:
                        class_id = int(parts[0])
                        if 0 <= class_id < NUM_CLASSES:
                            true_classes.append(class_id)
                        else:
                            print(f"Warning: Invalid class_id {class_id} in {label_file}, skipping")
                    except ValueError:
                        print(f"Warning: Invalid class_id (not an integer) in {label_file}: {line}, skipping")
        
        results = model.predict(img_path, conf=0.5)
        pred_classes = [int(cls) for result in results for cls in result.boxes.cls.cpu().numpy()]
        
        for cls in range(NUM_CLASSES):
            if cls in true_classes:
                y_true.append(cls)
                y_pred.append(cls if cls in pred_classes else NUM_CLASSES)
    
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES + 1))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES + ["None"], yticklabels=CLASSES + ["None"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(DATASET_ROOT, "confusion_matrix.png"))
    plt.close()

# # Streamlit App for Live Prediction
# def run_streamlit_app():
#     """Streamlit app for live predictions."""
#     st.title("Driver Monitoring System - Object Detection")
#     st.write("Upload an image or use webcam to detect driver behaviors/objects.")

#     model = YOLO(MODEL_PATH)
#     option = st.selectbox("Choose input method", ["Upload Image", "Webcam"])

#     if option == "Upload Image":
#         uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png"])
#         if uploaded_file is not None:
#             img = Image.open(uploaded_file).convert('RGB')
#             img_array = np.array(img)
#             results = model.predict(img_array, conf=0.5)
            
#             for result in results:
#                 boxes = result.boxes.xyxy.cpu().numpy()
#                 classes = result.boxes.cls.cpu().numpy()
#                 scores = result.boxes.conf.cpu().numpy()
                
#                 for box, cls, score in zip(boxes, classes, scores):
#                     x1, y1, x2, y2 = map(int, box)
#                     label = f"{CLASSES[int(cls)]} {score:.2f}"
#                     cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     cv2.putText(img_array, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
#             st.image(img_array, caption="Detected Objects", use_column_width=True)

#     elif option == "Webcam":
#         st.write("Webcam support requires local execution with OpenCV.")
#         cap = cv2.VideoCapture(0)
#         if not cap.isOpened():
#             st.error("Cannot access webcam.")
#             return
        
#         frame_placeholder = st.empty()
#         stop_button = st.button("Stop Webcam")
        
#         while not stop_button:
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             results = model.predict(frame, conf=0.5)
#             for result in results:
#                 boxes = result.boxes.xyxy.cpu().numpy()
#                 classes = result.boxes.cls.cpu().numpy()
#                 scores = result.boxes.conf.cpu().numpy()
                
#                 for box, cls, score in zip(boxes, classes, scores):
#                     x1, y1, x2, y2 = map(int, box)
#                     label = f"{CLASSES[int(cls)]} {score:.2f}"
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
#             frame_placeholder.image(frame, channels="BGR")
        
#         cap.release()

# # Main Execution
# def main():
#     """Main function to run the DMS YOLO project."""
#     validate_dataset()
#     prepare_data_yaml()
    
#     model, results = train_model(resume=False)
    
#     # Always run visualizations, even if model is loaded
#     test_dir = os.path.join(DATASET_ROOT, "test")
#     sample_img = os.path.join(test_dir, "images", os.listdir(os.path.join(test_dir, "images"))[0])
#     plot_bounding_boxes(model, sample_img, output_path=os.path.join(DATASET_ROOT, "bbox_image.png"))
#     plot_precision_recall(model, test_dir)
#     plot_confusion_matrix(model, test_dir)

#     st.write("Run `streamlit run dms_yolo_project.py` to start the web app.")

# if __name__ == "__main__":
#     main()


# # In[ ]:




