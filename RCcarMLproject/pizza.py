import cv2
from yolo_stuff.yolo_detector import YOLODetector
from send_udp_commands import send_udp_command

def detect_pizza():
    """
    Simple pizza detection using YOLO.
    When pizza is detected, sends forward command to RC car.
    """
    
    print("\n" + "="*60)
    print("PIZZA DETECTION WITH RC CAR CONTROL")
    print("="*60)
    print("\nInitializing YOLO detector...")
    
    yolo_detector = YOLODetector(
        weights_path="yolov4-tiny.weights",
        config_path="yolov4-tiny.cfg",
        names_path="coco.names",
        confidence_threshold=0.3,  # Lower threshold to catch pizza easier
        nms_threshold=0.4
    )
    
    if not yolo_detector.is_loaded():
        print("Failed to load YOLO model. Exiting.")
        return
    
    # RC car settings
    car_ip = "10.42.2.37"
    car_port = 8888
    
    # Connect to DroidCam
    droidcam_ip = "10.34.10.8"
    droidcam_port = 4747
    
    print(f"Connecting to DroidCam at {droidcam_ip}:{droidcam_port}...")
    cap = cv2.VideoCapture(f"http://{droidcam_ip}:{droidcam_port}/video")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("Failed to connect to DroidCam. Check IP and port.")
        return
    
    print("✓ Connected to DroidCam!")
    print(f"✓ Connected to RC car at {car_ip}:{car_port}\n")
    print("When PIZZA is detected, car will move FORWARD")
    print("When PIZZA disappears, car will STOP")
    print("Press 'q' to quit\n")
    
    frame_count = 0
    pizza_frames = 0
    last_command = "E"  # Track last command sent
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to get frame.")
            continue
        
        # Detect all objects
        detections = yolo_detector.detect(frame)
        
        frame_display = frame.copy()
        
        # Check for pizza
        pizza_detected = False
        
        for x, y, w, h, class_id, confidence in detections:
            color = yolo_detector.get_class_color(class_id)
            color = tuple(map(int, color))
            
            class_name = yolo_detector.get_class_name(class_id)
            label = f"{class_name} {confidence:.2f}"
            
            # Check if the class name contains "pizza"
            if "pizza" in class_name.lower():
                pizza_detected = True
                cv2.rectangle(frame_display, (x, y), (x + w, y + h), (0, 165, 255), 3)
                cv2.putText(frame_display, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            else:
                cv2.rectangle(frame_display, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame_display, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
        # Send command based on pizza detection
        if pizza_detected:
            command = "A"  # Move forward
            if command != last_command:
                send_udp_command(car_ip, car_port, command)
                last_command = command
            pizza_frames += 1
        else:
            command = "E"  # Stop
            if command != last_command:
                send_udp_command(car_ip, car_port, command)
                last_command = command
        
        # Draw info panel
        cv2.rectangle(frame_display, (0, 0), (frame_display.shape[1], 100), (0, 0, 0), -1)
        cv2.putText(frame_display, "PIZZA DETECTOR + RC CAR CONTROL", (10, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Status
        if pizza_detected:
            status_text = "✓ PIZZA DETECTED - CAR MOVING FORWARD"
            status_color = (0, 255, 0)
        else:
            status_text = "✗ No Pizza - CAR STOPPED"
            status_color = (0, 0, 255)
        
        cv2.putText(frame_display, status_text, (10, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(frame_display, f"Press Q to quit", 
                   (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.imshow("Pizza Detection + Car Control", frame_display)
        
        frame_count += 1
        
        if frame_count % 30 == 0:
            detection_rate = (pizza_frames / frame_count * 100) if frame_count > 0 else 0
            print(f"Frames: {frame_count} | Pizza detected: {pizza_frames} ({detection_rate:.1f}%) | Current command: {command}")
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    # Stop car when exiting
    send_udp_command(car_ip, car_port, "E")
    cap.release()
    cv2.destroyAllWindows()
    
    detection_rate = (pizza_frames / frame_count * 100) if frame_count > 0 else 0
    print(f"\n✓ Test complete!")
    print(f"Total frames: {frame_count}")
    print(f"Frames with pizza: {pizza_frames}")
    print(f"Detection rate: {detection_rate:.1f}%\n")

if __name__ == "__main__":
    detect_pizza()