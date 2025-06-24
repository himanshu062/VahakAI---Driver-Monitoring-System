import cv2
import torch
from ultralytics import YOLO
import time
from datetime import datetime
import os

class DMSDetector:
    def __init__(self):
        self.classes = ["Open Eye", "Closed Eye", "Cigarette", "Phone", "Seatbelt"]
        self.colors = {
            "Open Eye": (0, 255, 0),     # Green
            "Closed Eye": (0, 0, 255),   # Red
            "Cigarette": (255, 0, 0),    # Blue
            "Phone": (255, 165, 0),      # Orange
            "Seatbelt": (0, 255, 255)    # Yellow
        }
        self.save_dir = "detections"
        self.recording = False
        self.writer = None
        
        # Create save directory if it doesn't exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        # Load model
        print("Loading YOLO model...")
        self.model = YOLO("dms_yolo_model.pt")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

    def draw_detections(self, frame, results):
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Initialize detection counts
        detections = {cls: 0 for cls in self.classes}
        
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            
            for box, cls_idx, score in zip(boxes, classes, scores):
                x1, y1, x2, y2 = map(int, box)
                cls_name = self.classes[int(cls_idx)]
                color = self.colors[cls_name]
                
                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with confidence
                label = f"{cls_name} {score:.2f}"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(frame, (x1, y1 - text_size[1] - 4), (x1 + text_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Update detection count
                detections[cls_name] += 1
        
        # Draw detection summary
        y_pos = 30
        cv2.putText(frame, "Detections:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        for cls_name, count in detections.items():
            y_pos += 25
            color = self.colors[cls_name]
            cv2.putText(frame, f"{cls_name}: {count}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 2)
        
        # Add recording indicator if recording
        if self.recording:
            cv2.circle(frame, (width - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (width - 65, 35), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 0, 255), 2)
        
        return frame

    def toggle_recording(self, frame):
        if not self.recording:
            # Start recording
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.save_dir, f"dms_recording_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            frame_height, frame_width = frame.shape[:2]
            self.writer = cv2.VideoWriter(filename, fourcc, 20.0, (frame_width, frame_height))
            self.recording = True
            print(f"Started recording to {filename}")
        else:
            # Stop recording
            if self.writer:
                self.writer.release()
                self.writer = None
            self.recording = False
            print("Stopped recording")

    def run(self):
        print("Initializing camera...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        print("\nControls:")
        print("'q' - Quit")
        print("'r' - Toggle recording")
        print("'s' - Save current frame")
        
        fps_time = time.time()
        fps_counter = 0
        fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame")
                break

            # Run detection
            results = self.model(frame, conf=0.5)
            
            # Draw detections and info
            frame = self.draw_detections(frame, results)
            
            # Calculate and display FPS
            fps_counter += 1
            if time.time() - fps_time > 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_time = time.time()
            cv2.putText(frame, f"FPS: {fps}", (10, frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Display frame
            cv2.imshow("DMS Detection", frame)
            
            # Save frame if recording
            if self.recording and self.writer:
                self.writer.write(frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.toggle_recording(frame)
            elif key == ord('s'):
                # Save current frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(self.save_dir, f"dms_frame_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                print(f"Saved frame to {filename}")

        # Cleanup
        if self.writer:
            self.writer.release()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = DMSDetector()
    detector.run()