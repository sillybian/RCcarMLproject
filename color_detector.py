import cv2
import numpy as np

class OrangeBlockDetector:
    """
    Detects neon orange Jenga blocks using HSV color range detection.
    Supports real-time HSV parameter adjustment.
    """
    
    def __init__(self, lower_h=0, lower_s=210, lower_v=100, 
                 upper_h=25, upper_s=255, upper_v=255, min_area=500):
        """
        Initialize the orange block detector.
        
        Args:
            lower_h, lower_s, lower_v: Lower HSV bounds
            upper_h, upper_s, upper_v: Upper HSV bounds
            min_area: Minimum contour area to be considered a block
        """
        self.lower_h = lower_h
        self.lower_s = lower_s
        self.lower_v = lower_v
        self.upper_h = upper_h
        self.upper_s = upper_s
        self.upper_v = upper_v
        self.min_area = min_area
        
        # Create morphological kernel for cleaning
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    def detect(self, frame):
        """
        Detect orange blocks in a frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            detections: List of (x, y, w, h) tuples for each detected block
            mask: The processed mask used for detection
        """
        if frame is None:
            return [], None
        
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create lower and upper bounds
        lower_orange = np.array([self.lower_h, self.lower_s, self.lower_v])
        upper_orange = np.array([self.upper_h, self.upper_s, self.upper_v])
        
        # Create mask for orange color
        mask = cv2.inRange(hsv, lower_orange, upper_orange)
        
        # Apply morphological operations to clean up the mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area:
                x, y, w, h = cv2.boundingRect(contour)
                detections.append((x, y, w, h))
        
        return detections, mask
    
    def set_hsv_range(self, lower_h, lower_s, lower_v, upper_h, upper_s, upper_v):
        """
        Update the HSV color range for detection.
        
        Args:
            lower_h, lower_s, lower_v: Lower HSV bounds
            upper_h, upper_s, upper_v: Upper HSV bounds
        """
        self.lower_h = lower_h
        self.lower_s = lower_s
        self.lower_v = lower_v
        self.upper_h = upper_h
        self.upper_s = upper_s
        self.upper_v = upper_v
    
    def set_min_area(self, min_area):
        """Update the minimum block area threshold."""
        self.min_area = min_area
    
    def get_hsv_range(self):
        """Get current HSV range settings."""
        return {
            'lower_h': self.lower_h,
            'lower_s': self.lower_s,
            'lower_v': self.lower_v,
            'upper_h': self.upper_h,
            'upper_s': self.upper_s,
            'upper_v': self.upper_v,
            'min_area': self.min_area
        }


def test_color_detector():
    """
    Interactive color detector test with real-time HSV adjustment.
    Use trackbars to fine-tune the orange detection.
    """
    print("\n" + "="*60)
    print("ORANGE BLOCK DETECTOR - HSV TUNING")
    print("="*60)
    print("\nInitializing detector...")
    
    detector = OrangeBlockDetector(
        lower_h=5, lower_s=210, lower_v=100,
        upper_h=25, upper_s=255, upper_v=255,
        min_area=500
    )
    
    # DroidCam settings
    droidcam_ip = "10.34.13.69"
    droidcam_port = 4747
    
    print(f"Connecting to DroidCam at {droidcam_ip}:{droidcam_port}...")
    cap = cv2.VideoCapture(f"http://{droidcam_ip}:{droidcam_port}/video")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("Failed to connect to DroidCam. Check IP and port.")
        return
    
    print("✓ Connected to DroidCam!")
    print("\nControls:")
    print("  Use trackbars to adjust HSV range")
    print("  P     - Print current settings to console")
    print("  S     - Save settings to file")
    print("  Q     - Quit\n")
    
    # Create window
    window_name = "Orange Block Detector - HSV Tuning"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Create trackbars for lower HSV bounds
    cv2.createTrackbar("Lower H", window_name, detector.lower_h, 180, lambda x: None)
    cv2.createTrackbar("Lower S", window_name, detector.lower_s, 255, lambda x: None)
    cv2.createTrackbar("Lower V", window_name, detector.lower_v, 255, lambda x: None)
    
    # Create trackbars for upper HSV bounds
    cv2.createTrackbar("Upper H", window_name, detector.upper_h, 180, lambda x: None)
    cv2.createTrackbar("Upper S", window_name, detector.upper_s, 255, lambda x: None)
    cv2.createTrackbar("Upper V", window_name, detector.upper_v, 255, lambda x: None)
    
    # Create trackbar for min area
    cv2.createTrackbar("Min Area", window_name, detector.min_area, 10000, lambda x: None)
    
    frame_count = 0
    blocks_detected = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to get frame.")
            continue
        
        # Get trackbar values
        lower_h = cv2.getTrackbarPos("Lower H", window_name)
        lower_s = cv2.getTrackbarPos("Lower S", window_name)
        lower_v = cv2.getTrackbarPos("Lower V", window_name)
        upper_h = cv2.getTrackbarPos("Upper H", window_name)
        upper_s = cv2.getTrackbarPos("Upper S", window_name)
        upper_v = cv2.getTrackbarPos("Upper V", window_name)
        min_area = cv2.getTrackbarPos("Min Area", window_name)
        
        # Update detector with trackbar values
        detector.set_hsv_range(lower_h, lower_s, lower_v, upper_h, upper_s, upper_v)
        detector.set_min_area(min_area)
        
        # Detect blocks
        detections, mask = detector.detect(frame)
        blocks_detected = len(detections)
        
        # Draw detections on original frame
        frame_display = frame.copy()
        for x, y, w, h in detections:
            cv2.rectangle(frame_display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame_display, "Orange Block", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add info panel to frame
        height, width = frame_display.shape[:2]
        cv2.rectangle(frame_display, (0, 0), (width, 200), (0, 0, 0), -1)
        
        cv2.putText(frame_display, "ORANGE BLOCK DETECTOR - HSV TUNING", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display current HSV values
        cv2.putText(frame_display, f"Lower: H={lower_h:3d} S={lower_s:3d} V={lower_v:3d}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1)
        cv2.putText(frame_display, f"Upper: H={upper_h:3d} S={upper_s:3d} V={upper_v:3d}", (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1)
        cv2.putText(frame_display, f"Min Area: {min_area}", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1)
        
        # Display block count
        cv2.putText(frame_display, f"Blocks detected: {blocks_detected}", (10, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 165), 2)
        
        # Display instructions
        cv2.putText(frame_display, "Use trackbars to adjust | P: Print | S: Save | Q: Quit", (10, 170),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame_display, f"Frame: {frame_count}", (10, 195),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Display both the original frame with detections and the mask
        cv2.imshow(window_name, frame_display)
        cv2.imshow("Orange Mask", mask)
        
        frame_count += 1
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            # Print current settings
            settings = detector.get_hsv_range()
            print(f"\n{'='*60}")
            print("CURRENT HSV SETTINGS")
            print(f"{'='*60}")
            print(f"lower_h={settings['lower_h']}, lower_s={settings['lower_s']}, lower_v={settings['lower_v']},")
            print(f"upper_h={settings['upper_h']}, upper_s={settings['upper_s']}, upper_v={settings['upper_v']},")
            print(f"min_area={settings['min_area']}")
            print(f"{'='*60}\n")
        elif key == ord('s'):
            # Save settings to file
            settings = detector.get_hsv_range()
            with open('hsv_settings.txt', 'w') as f:
                f.write(f"# Orange Block Detector - HSV Settings\n")
                f.write(f"# Generated from frame {frame_count}\n\n")
                f.write(f"OrangeBlockDetector(\n")
                f.write(f"    lower_h={settings['lower_h']},\n")
                f.write(f"    lower_s={settings['lower_s']},\n")
                f.write(f"    lower_v={settings['lower_v']},\n")
                f.write(f"    upper_h={settings['upper_h']},\n")
                f.write(f"    upper_s={settings['upper_s']},\n")
                f.write(f"    upper_v={settings['upper_v']},\n")
                f.write(f"    min_area={settings['min_area']}\n")
                f.write(f")\n")
            print(f"\n✓ Settings saved to hsv_settings.txt")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final settings
    settings = detector.get_hsv_range()
    print(f"\n{'='*60}")
    print("FINAL HSV SETTINGS")
    print(f"{'='*60}")
    print(f"Copy and paste these settings into your code:\n")
    print(f"OrangeBlockDetector(")
    print(f"    lower_h={settings['lower_h']},")
    print(f"    lower_s={settings['lower_s']},")
    print(f"    lower_v={settings['lower_v']},")
    print(f"    upper_h={settings['upper_h']},")
    print(f"    upper_s={settings['upper_s']},")
    print(f"    upper_v={settings['upper_v']},")
    print(f"    min_area={settings['min_area']}")
    print(f")\n")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    test_color_detector()