import cv2
import numpy as np
import math

class CarCalibrator:
    """
    Calibrates the RC car by manually marking the pizza position and orientation.
    """
    
    def __init__(self):
        self.pizza_center = None
        self.pizza_point = None
        self.calibration_angle = None
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks to mark pizza center and point"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.pizza_center is None:
                self.pizza_center = (x, y)
                print(f"✓ Pizza center marked at ({x}, {y})")
            elif self.pizza_point is None:
                self.pizza_point = (x, y)
                self.calculate_angle()
                print(f"✓ Pizza point marked at ({x}, {y})")
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click to reset
            self.pizza_center = None
            self.pizza_point = None
            self.calibration_angle = None
            print("✓ Marks cleared - click again to set new points")
    
    def calculate_angle(self):
        """Calculate the angle from center to point"""
        if self.pizza_center and self.pizza_point:
            center_x, center_y = self.pizza_center
            point_x, point_y = self.pizza_point
            
            dx = point_x - center_x
            dy = point_y - center_y
            self.calibration_angle = math.degrees(math.atan2(dy, dx)) % 360

def calibrate():
    print("\n" + "="*70)
    print("RC CAR CALIBRATION TOOL (MANUAL)")
    print("="*70)
    print("\nThis tool helps establish where the pizza marker's point is facing.")
    print("\nSetup:")
    print("1. Place pizza slice on top of the RC car")
    print("2. Position the car pointing in a known direction (e.g., north/up)")
    print("3. The camera will show the top-down view of the car")
    print("\nHOW TO USE:")
    print("  LEFT CLICK   - Mark pizza center (first click)")
    print("  LEFT CLICK   - Mark pizza point (second click)")
    print("  RIGHT CLICK  - Clear marks and start over")
    print("  C            - Capture this angle (saves it)")
    print("  S            - Save & calculate average from all captures")
    print("  Q            - Quit without saving")
    print("\nMake sure to click on the window to give it focus!")
    print("="*70 + "\n")
    
    calibrator = CarCalibrator()
    
    # Connect to DroidCam
    droidcam_ip = "10.34.10.8"  # Replace with your IP if different
    droidcam_port = 4747
    
    print(f"Connecting to DroidCam at {droidcam_ip}:{droidcam_port}...")
    cap = cv2.VideoCapture(f"http://{droidcam_ip}:{droidcam_port}/video")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("Failed to connect to DroidCam. Check IP and port.")
        return
    
    print("✓ Connected to DroidCam\n")
    
    calibration_angles = []
    
    # Create window and set mouse callback
    window_name = "Car Calibration - Click to mark pizza"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, calibrator.mouse_callback)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to get frame.")
            continue
        
        frame_display = frame.copy()
        
        # Draw status panel
        cv2.rectangle(frame_display, (0, 0), (frame_display.shape[1], 200), (20, 20, 20), -1)
        
        cv2.putText(frame_display, "RC CAR CALIBRATION (MANUAL)", (10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        
        # Instructions
        instructions = [
            "1. LEFT CLICK on pizza CENTER",
            "2. LEFT CLICK on pizza POINT (tip)",
            "3. RIGHT CLICK to reset"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame_display, instruction, (10, 75 + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
        
        # Draw marked points
        if calibrator.pizza_center:
            x, y = calibrator.pizza_center
            cv2.circle(frame_display, (x, y), 10, (0, 255, 0), -1)
            cv2.putText(frame_display, "CENTER", (x + 15, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if calibrator.pizza_point:
            x, y = calibrator.pizza_point
            cv2.circle(frame_display, (x, y), 10, (0, 0, 255), -1)
            cv2.putText(frame_display, "POINT", (x + 15, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw line between center and point
        if calibrator.pizza_center and calibrator.pizza_point:
            center_x, center_y = calibrator.pizza_center
            point_x, point_y = calibrator.pizza_point
            cv2.line(frame_display, (center_x, center_y), (point_x, point_y), (255, 255, 0), 3)
            
            # Show angle
            angle_text = f"Angle: {calibrator.calibration_angle:.1f}°"
            cv2.putText(frame_display, angle_text, (10, frame_display.shape[0] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Show captured angles
        if calibration_angles:
            avg_angle = sum(calibration_angles) / len(calibration_angles)
            angles_text = f"Captured: {len(calibration_angles)} | Avg: {avg_angle:.1f}°"
            cv2.putText(frame_display, angles_text, (10, frame_display.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Instructions at bottom
        cv2.putText(frame_display, "C=Capture | S=Save | Q=Quit",
                   (frame_display.shape[1] - 300, frame_display.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
        
        cv2.imshow(window_name, frame_display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            print("\n✗ Calibration cancelled.\n")
            break
        elif key == ord('c') or key == ord('C'):
            if calibrator.calibration_angle is not None:
                calibration_angles.append(calibrator.calibration_angle)
                print(f"✓ Captured angle #{len(calibration_angles)}: {calibrator.calibration_angle:.1f}°")
            else:
                print("✗ Please mark pizza center and point first!")
        elif key == ord('s') or key == ord('S'):
            if calibration_angles:
                avg_angle = sum(calibration_angles) / len(calibration_angles)
                std_dev = np.std(calibration_angles)
                print(f"\n" + "="*70)
                print(f"✓ CALIBRATION SAVED!")
                print(f"="*70)
                print(f"Number of samples:        {len(calibration_angles)}")
                print(f"Average pizza angle:      {avg_angle:.2f}°")
                print(f"Standard deviation:       {std_dev:.2f}°")
                print(f"Min angle:                {min(calibration_angles):.2f}°")
                print(f"Max angle:                {max(calibration_angles):.2f}°")
                print(f"="*70)
                print("\nAdd this to your main.py:")
                print(f"car_front_angle_offset = {avg_angle:.2f}  # Pizza point angle when car faces 'forward'\n")
                print("This means your car's front is at this angle in pixel coordinates.")
                print("="*70 + "\n")
                break
            else:
                print("✗ No angles captured yet! Press 'C' after marking points.")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    calibrate()