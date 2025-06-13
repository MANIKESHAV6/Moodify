import cv2
import os
from deepface import DeepFace
import time

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("🎯 Simple Mood Detector - Debug Version")
print("=" * 40)

# Test camera first
print("1. Testing camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not found! Trying other camera indices...")
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"✅ Found camera at index {i}")
            break
    else:
        print("❌ No camera found!")
        exit()

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Test frame capture
ret, test_frame = cap.read()
if ret:
    print(f"✅ Camera working! Frame size: {test_frame.shape}")
else:
    print("❌ Cannot capture frames!")
    exit()

# Test DeepFace
print("2. Testing DeepFace...")
try:
    # Use a small portion of the test frame
    small_face = cv2.resize(test_frame[100:300, 100:300], (224, 224))
    result = DeepFace.analyze(small_face, actions=['emotion'], enforce_detection=False, silent=True)
    print("✅ DeepFace working!")
except Exception as e:
    print(f"❌ DeepFace error: {e}")
    print("Installing required models...")

print("3. Starting emotion detection...")
print("Controls: 'q' to quit, 'space' for manual detection")
print("=" * 40)

# Simple face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

frame_count = 0
last_detection = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame")
        break
    
    frame_count += 1
    current_time = time.time()
    
    # Detect faces every frame (for debugging)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))
    
    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"Face detected {w}x{h}", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Try emotion detection every 3 seconds or on spacebar
        if (current_time - last_detection > 3.0) or cv2.waitKey(1) & 0xFF == ord(' '):
            print(f"Analyzing face {w}x{h}...")
            
            try:
                # Extract face region
                face_roi = frame[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi, (224, 224))
                
                # Analyze emotion
                result = DeepFace.analyze(
                    face_resized, 
                    actions=['emotion'], 
                    enforce_detection=False, 
                    silent=True
                )
                
                if result:
                    emotions = result[0]['emotion'] if isinstance(result, list) else result['emotion']
                    dominant_emotion = max(emotions, key=emotions.get)
                    confidence = emotions[dominant_emotion]
                    
                    print(f"🎭 Emotion: {dominant_emotion} ({confidence:.1f}%)")
                    
                    # Display on frame
                    color = (0, 255, 0) if confidence > 50 else (0, 165, 255)
                    cv2.putText(frame, f"{dominant_emotion} {confidence:.0f}%", 
                               (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    last_detection = current_time
                
            except Exception as e:
                print(f"❌ Emotion detection failed: {e}")
                cv2.putText(frame, "Detection failed", (x, y+h+30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Show frame info
    cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Faces: {len(faces)}", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Display frame
    cv2.imshow('Simple Mood Detector', frame)
    
    # Check for quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

print("Cleaning up...")
cap.release()
cv2.destroyAllWindows()
print("Done!")