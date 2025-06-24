import os
import cv2
import numpy as np
from ultralytics import YOLO

# Configuration
DATASET_ROOT = r"C:\Users\anshg\Documents\final project"
MODEL_PATH = os.path.join(DATASET_ROOT, "dms_yolo_model.pt")
TEST_IMAGES_DIR = os.path.join(DATASET_ROOT, "test", "images")
OUTPUT_DIR = os.path.join(DATASET_ROOT, "output_images")
CLASSES = ["Open Eye", "Closed Eye", "Cigarette", "Phone", "Seatbelt"]

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Load the model
model = YOLO(MODEL_PATH)
print(f"Loaded model from {MODEL_PATH}")

# Function to predict and save results for a single image
def predict_and_save_image(image_path, output_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    # Perform prediction
    results = model.predict(image_path, conf=0.5)
    
    # Draw bounding boxes on the image
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        
        for box, cls, score in zip(boxes, classes, scores):
            x1, y1, x2, y2 = map(int, box)
            label = f"{CLASSES[int(cls)]} {score:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save the output image
    cv2.imwrite(output_path, img)
    print(f"Saved output image with detections at {output_path}")

# Main execution
if __name__ == "__main__":
    # Get all images in the test/images directory
    image_files = [f for f in os.listdir(TEST_IMAGES_DIR) if f.endswith(('.jpg', '.png'))]
    
    if not image_files:
        print(f"No images found in {TEST_IMAGES_DIR}")
    else:
        print(f"Found {len(image_files)} images to process")
        
        # Process each image
        for image_file in image_files:
            image_path = os.path.join(TEST_IMAGES_DIR, image_file)
            output_path = os.path.join(OUTPUT_DIR, f"output_{image_file}")
            predict_and_save_image(image_path, output_path)