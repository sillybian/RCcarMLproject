import cv2
import numpy as np

# Test with a video file or live camera
# Change this to your DroidCam URL
cap = cv2.VideoCapture("http://10.34.10.8:4747/video")  # Change IP to yours
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Failed to connect to camera. Check your IP address.")
    print("Make sure DroidCam is running and you have the correct IP.")
    exit()

print("Starting color detection test...")

print("Adjust the HSV ranges below and re-run the script to test different values.")
print("Press 'q' to quit\n")

# Default orange color range (you'll adjust these)
lower_h = 5
lower_s = 200
lower_v = 100
upper_h = 25
upper_s = 255
upper_v = 255

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame. Check connection.")
        break
    
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create mask with current color range
    lower_orange = np.array([lower_h, lower_s, lower_v])
    upper_orange = np.array([upper_h, upper_s, upper_v])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on original frame
    frame_with_contours = frame.copy()
    cv2.drawContours(frame_with_contours, contours, -1, (0, 165, 255), 2)
    
    # Add text showing current range
    cv2.putText(frame_with_contours, f"H: {lower_h}-{upper_h}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame_with_contours, f"S: {lower_s}-{upper_s}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame_with_contours, f"V: {lower_v}-{upper_v}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame_with_contours, f"Blocks detected: {len(contours)}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display images
    cv2.imshow('Original with Contours', frame_with_contours)
    cv2.imshow('Color Mask', mask)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\nTo adjust detection:")
print("1. Edit these values in calibrate_color.py (lines 17-23):")
print(f"   lower_h = {lower_h}")
print(f"   lower_s = {lower_s}")
print(f"   lower_v = {lower_v}")
print(f"   upper_h = {upper_h}")
print(f"   upper_s = {upper_s}")
print(f"   upper_v = {upper_v}")
print("\n2. Then copy these exact values to main_color_detection.py (lines 12-13)")