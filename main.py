import cv2
import numpy as np
from color_detector import OrangeBlockDetector
from yolo_stuff.yolo_detector import YOLODetector
from pathfinder import GridPathfinder
from send_udp_commands import send_udp_command

class RCCarController:
    def __init__(self):
        self.selected_goal = None
        self.is_running = False
        self.frame = None
        self.car_position = None  # Will store pizza/car position
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks to select goal position"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_goal = (x, y)
            print(f"Goal selected at pixel position: ({x}, {y})")

def draw_detections(frame, color_detections, yolo_detections, yolo_detector):
    """
    Draw bounding boxes for orange blocks.
    """
    # Draw orange block detections
    for x, y, w, h in color_detections:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
        cv2.putText(frame, "Jenga Block", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
    
    # Draw YOLO detections (pizza/car marker)
    for x, y, w, h, class_id, confidence in yolo_detections:
        color = yolo_detector.get_class_color(class_id)
        color = tuple(map(int, color))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        class_name = yolo_detector.get_class_name(class_id)
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame

def find_car_marker(yolo_detections, yolo_detector):
    """
    Find the pizza slice marker on the RC car (class 30 in COCO dataset).
    This tells us where the car is and which direction it's facing.
    
    Returns:
        (x, y, w, h) of pizza/car marker, or None if not found
    """
    for x, y, w, h, class_id, confidence in yolo_detections:
        if class_id == 30:  # Pizza class in COCO - used as car marker
            return (x, y, w, h)
    return None

def draw_ui_info(frame, is_running, selected_goal, car_position):
    """
    Draw UI information on the frame.
    """
    # Status bar background
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 120), (0, 0, 0), -1)
    
    # Status text
    status_text = "RUNNING" if is_running else "STOPPED"
    status_color = (0, 255, 0) if is_running else (0, 0, 255)
    cv2.putText(frame, f"Status: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    
    # Car position info
    if car_position:
        cv2.putText(frame, f"Car: ({car_position[0]}, {car_position[1]})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    else:
        cv2.putText(frame, "Car: NOT DETECTED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Goal info
    if selected_goal:
        cv2.putText(frame, f"Goal: ({selected_goal[0]}, {selected_goal[1]})", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    else:
        cv2.putText(frame, "Goal: Click to select", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Instructions
    cv2.putText(frame, "SPACE: Start/Stop | CLICK: Set Goal | Q: Quit", 
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return frame

def main():
    # DroidCam settings
    droidcam_ip = "10.34.10.8"  # Replace with your DroidCam IP
    droidcam_port = 4747
    
    # RC car settings
    car_ip = "10.42.4.140"
    car_port = 8888
    
    # Initialize controller
    controller = RCCarController()
    
    # Initialize orange block detector (color-based)
    print("Initializing orange block detector...")
    color_detector = OrangeBlockDetector(
        lower_h=5, lower_s=200, lower_v=100,
        upper_h=25, upper_s=255, upper_v=255,
        min_area=500
    )
    
    # Initialize YOLO detector (for pizza marker on car)
    print("Initializing YOLO detector...")
    yolo_detector = YOLODetector(
        weights_path="yolov4.weights",
        config_path="yolov4.cfg",
        names_path="coco.names",
        confidence_threshold=0.5,
        nms_threshold=0.4
    )
    
    if not yolo_detector.is_loaded():
        print("Failed to load YOLO model. Exiting.")
        return
    
    # Initialize pathfinder
    print("Initializing pathfinder...")
    pathfinder = GridPathfinder(
        grid_width=50,
        grid_height=50,
        car_radius=3,      # Keep this much space around the car
        wall_padding=2     # Extra padding around obstacles
    )
    
    # Initialize camera
    print(f"Connecting to DroidCam at {droidcam_ip}:{droidcam_port}...")
    cap = cv2.VideoCapture(f"http://{droidcam_ip}:{droidcam_port}/video")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("Failed to connect to DroidCam. Check IP and port.")
        return
    
    print("\n" + "="*50)
    print("RC CAR CONTROLLER")
    print("="*50)
    print("Controls:")
    print("  SPACE  - Start/Stop the car")
    print("  CLICK  - Click on screen to set destination")
    print("  Q      - Quit")
    print("="*50 + "\n")
    
    frame_count = 0
    car_grid_x = 25  # Default starting position in grid
    car_grid_y = 25
    
    # Create window and set mouse callback
    window_name = "RC Car Controller"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, controller.mouse_callback)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to get frame. Retrying...")
            continue
        
        frame_height, frame_width = frame.shape[:2]
        
        # Detect orange blocks (color-based)
        color_detections, mask = color_detector.detect(frame)
        
        # Detect pizza marker on car with YOLO
        yolo_detections = yolo_detector.detect(frame)
        
        # Find car position from pizza marker
        car_marker = find_car_marker(yolo_detections, yolo_detector)
        
        # Build grid map from detections (orange blocks only)
        pathfinder.build_grid(frame, color_detections, yolo_detections)
        
        command = "E"  # Default: stop
        
        # Update car position if detected
        if car_marker:
            car_x, car_y, car_w, car_h = car_marker
            controller.car_position = (car_x + car_w // 2, car_y + car_h // 2)
            
            # Convert to grid coordinates
            car_grid_x = int((car_x + car_w / 2) * (pathfinder.grid_width / frame_width))
            car_grid_y = int((car_y + car_h / 2) * (pathfinder.grid_height / frame_height))
        
        # Only pathfind if car is running and goal is selected
        if controller.is_running and controller.selected_goal:
            goal_pixel_x, goal_pixel_y = controller.selected_goal
            goal_grid_x = int(goal_pixel_x * (pathfinder.grid_width / frame_width))
            goal_grid_y = int(goal_pixel_y * (pathfinder.grid_height / frame_height))
            
            # Find path to goal
            path = pathfinder.find_path(car_grid_x, car_grid_y, goal_grid_x, goal_grid_y)
            
            if path and len(path) > 1:
                waypoint = pathfinder.get_next_waypoint(car_grid_x, car_grid_y)
                
                if waypoint:
                    command = pathfinder.get_direction_to_waypoint(
                        car_grid_x, car_grid_y, waypoint[0], waypoint[1]
                    )
            else:
                if controller.selected_goal:
                    print("No path to goal found!")
                command = "E"  # Stop if no path
        
        # Draw all detections on frame
        frame_display = draw_detections(frame.copy(), color_detections, yolo_detections, yolo_detector)
        
        # Draw grid visualization
        grid_visualization = pathfinder.visualize_grid(frame_display)
        
        # Draw UI info
        grid_visualization = draw_ui_info(grid_visualization, controller.is_running, 
                                          controller.selected_goal, controller.car_position)
        
        # Draw selected goal marker if set
        if controller.selected_goal:
            cv2.circle(grid_visualization, controller.selected_goal, 10, (0, 255, 255), 3)
            cv2.putText(grid_visualization, "GOAL", (controller.selected_goal[0] - 20, controller.selected_goal[1] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Send command to RC car (only if running)
        if controller.is_running:
            send_udp_command(car_ip, car_port, command)
        else:
            # Send stop command when not running
            send_udp_command(car_ip, car_port, "E")
        
        # Display frames
        cv2.imshow("Detection with Detections", frame_display)
        cv2.imshow(window_name, grid_visualization)
        cv2.imshow("Color Mask", mask)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Frames: {frame_count} | Status: {'RUNNING' if controller.is_running else 'STOPPED'}")
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Spacebar
            controller.is_running = not controller.is_running
            status = "STARTED" if controller.is_running else "STOPPED"
            print(f"Car {status}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Send stop command to RC car
    send_udp_command(car_ip, car_port, "E")
    print("Shutting down...")

if __name__ == "__main__":
    main()