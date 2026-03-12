# webcam_pill_detector.py  ← Run this ON YOUR LAPTOP
# First install: pip install opencv-python tensorflow numpy matplotlib

import cv2
import numpy as np
import tensorflow as tf
import json

# ============================================================
# CONFIGURATION - Update these paths
# ============================================================
MODEL_PATH = 'pill_model_final.h5'       # Download from Google Drive
CLASS_NAMES_PATH = 'class_names.json'    # Download from Google Drive
# ============================================================

class WebcamPillDetector:
    def __init__(self, model_path, class_names_path):
        print("Loading model...")
        self.model = tf.keras.models.load_model(model_path)
        
        with open(class_names_path) as f:
            self.class_names = json.load(f)
        
        self.img_size = (224, 224)
        self.current_result = None
        print(f"✅ Model loaded! {len(self.class_names)} classes.")
    
    def preprocess(self, frame):
        img = cv2.resize(frame, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)
    
    def segment_pill(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask
    
    def get_shape(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return "unknown", None
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < 1000:
            return "too_small", None
        peri = cv2.arcLength(c, True)
        if peri == 0:
            return "unknown", c
        circularity = (4 * np.pi * area) / (peri ** 2)
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / h if h > 0 else 1
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        v = len(approx)
        
        if circularity > 0.85: shape = "round"
        elif circularity > 0.70: shape = "oval" if aspect_ratio > 1.2 else "round"
        elif aspect_ratio > 2.2: shape = "oblong"
        elif aspect_ratio > 1.4: shape = "capsule"
        elif v == 3: shape = "triangle"
        elif v == 4: shape = "square" if 0.9 <= aspect_ratio <= 1.1 else "rectangle"
        elif v == 5: shape = "pentagon"
        elif v == 6: shape = "hexagon"
        else: shape = "round"
        return shape, c
    
    def get_color(self, frame, mask):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        pixels = hsv[mask > 0]
        if len(pixels) == 0:
            return "unknown"
        h, s, v = np.mean(pixels[:,0]), np.mean(pixels[:,1]), np.mean(pixels[:,2])
        if s < 30:
            if v > 200: return "white"
            elif v > 150: return "off_white"
            elif v > 80: return "gray"
            else: return "black"
        if h < 10 or h > 170: return "red" if s > 100 else "pink"
        elif h < 25: return "orange"
        elif h < 35: return "yellow"
        elif h < 80: return "green"
        elif h < 130: return "blue"
        elif h < 160: return "purple"
        else: return "pink"
    
    def predict(self, frame):
        model_input = self.preprocess(frame)
        preds = self.model.predict(model_input, verbose=0)[0]
        top3 = np.argsort(preds)[::-1][:3]
        return [(self.class_names[i], float(preds[i])) for i in top3]
    
    def draw_overlay(self, frame, predictions, shape, color, contour):
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw contour
        if contour is not None:
            cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 3)
        
        # Draw semi-transparent panel
        panel = frame.copy()
        cv2.rectangle(panel, (0, 0), (w, 160), (0, 0, 0), -1)
        frame = cv2.addWeighted(panel, 0.6, frame, 0.4, 0)
        
        # Instructions
        cv2.putText(frame, "Place pill in frame | Press Q to quit | S to save",
                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Results
        cv2.putText(frame, "PILL ANALYSIS", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.putText(frame, f"Shape: {shape} | Color: {color}", (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        
        # Top 3 predictions with confidence bars
        for i, (cls, conf) in enumerate(predictions):
            y_pos = 85 + i * 25
            bar_width = int(conf * 200)
            color_bar = (0, 255, 0) if i == 0 else (100, 200, 100)
            cv2.rectangle(frame, (10, y_pos - 15), (10 + bar_width, y_pos), color_bar, -1)
            cv2.putText(frame, f"{cls}: {conf*100:.1f}%", (15, y_pos - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Overlay contour on top
        if contour is not None:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        
        return frame
    
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        
        if not cap.isOpened():
            print("❌ Could not open webcam!")
            return
        
        print("🎥 Webcam started! Press Q to quit, S to save screenshot.")
        
        frame_count = 0
        predictions = []
        shape = "detecting..."
        color = "detecting..."
        contour = None
        save_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Analyze every 10 frames (for performance)
            if frame_count % 10 == 0:
                # Resize working copy
                work = cv2.resize(frame, (400, 400))
                
                # Segment
                mask = self.segment_pill(work)
                
                # Get shape and contour
                shape, raw_contour = self.get_shape(mask)
                
                # Scale contour back to display size
                if raw_contour is not None:
                    h, w = frame.shape[:2]
                    scale_x = w / 400
                    scale_y = h / 400
                    contour = (raw_contour * [scale_x, scale_y]).astype(np.int32)
                
                # Get color
                color = self.get_color(work, mask)
                
                # Deep learning prediction
                predictions = self.predict(frame)
            
            # Draw overlay
            display = self.draw_overlay(frame.copy(), predictions, shape, color, contour)
            
            cv2.imshow("💊 Pill Identification System - KDU Group 10", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                save_path = f"pill_capture_{save_count}.jpg"
                cv2.imwrite(save_path, display)
                print(f"📸 Saved: {save_path}")
                save_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Webcam closed.")


# Run the detector
detector = WebcamPillDetector(MODEL_PATH, CLASS_NAMES_PATH)
detector.run()