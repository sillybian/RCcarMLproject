"""
Interactive RC Car Navigator with DroidCam Live Feed and Click-to-Navigate

This module provides an interactive interface for controlling the RC car:
- Live video feed from DroidCam on your laptop screen
- Click on the screen to set navigation goals
- Automatic pathfinding around neon orange Jenga block obstacles
- Real-time visualization of the grid map and detected obstacles
"""

import cv2
import numpy as np
from color_detector import OrangeBlockDetector
from yolo_detector import YOLODetector
from pathfinder import GridPathfinder
from send_udp_commands import send_udp_command
import threading
import time


class InteractiveNavigator:
    def __init__(self, droidcam_ip, droidcam_port, car_ip, car_port):
        """
        Initialize the interactive navigator.
        
        Args:
            droidcam_ip: IP address of device running DroidCam
            droidcam_port: Port number for DroidCam (default 4747)
            car_ip: IP address of RC car
            car_port: Port of RC car UDP listener
        """
        self.droidcam_ip = droidcam_ip
        self.droidcam_port = droidcam_port
        self.car_ip = car_ip
        self.car_port = car_port
        
        # State variables
        self.selected_goal = None
        self.is_running = False
        self.current_frame = None
        self.car_position = None
        self.last_command = "E"
        self.path_in_progress = False
        self.arrival_threshold = 10  # pixels
        
        # Detectors and pathfinder
        self.color_detector = None
        self.yolo_detector = None
        self.pathfinder = None
        self.video_capture = None
        
        # Threading
        self.video_thread = None
        self.control_thread = None
        self.stop_flag = False
        self.frame_lock = threading.Lock()
        
    def setup_detectors(self):
        """Initialize color detector, YOLO detector, and pathfinder."""
        print("[SETUP] Initializing orange block detector...")
        self.color_detector = OrangeBlockDetector(
            lower_h=5, lower_s=200, lower_v=100,
            upper_h=25, upper_s=255, upper_v=255,
            min_area=500
        )
        
        print("[SETUP] Initializing YOLO detector...")
        self.yolo_detector = YOLODetector(
            weights_path="yolov4.weights",
            config_path="yolov4.cfg",
            names_path="coco.names",
            confidence_threshold=0.5,
            nms_threshold=0.4
        )
        
        if not self.yolo_detector.is_loaded():
            raise RuntimeError("Failed to load YOLO model")
        
        print("[SETUP] Initializing pathfinder...")
        self.pathfinder = GridPathfinder(
            grid_width=50,
            grid_height=50,
            car_radius=3,
            wall_padding=2
        )
        
        print("[SETUP] ✓ All detectors initialized successfully")
    
    def connect_camera(self):
        """Establish connection to DroidCam."""
        print(f"[CAMERA] Connecting to DroidCam at {self.droidcam_ip}:{self.droidcam_port}...")
        url = f"http://{self.droidcam_ip}:{self.droidcam_port}/video"
        self.video_capture = cv2.VideoCapture(url)
        self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self.video_capture.isOpened():
            raise RuntimeError("Failed to connect to DroidCam. Check IP and port.")
        
        print("[CAMERA] ✓ Connected to DroidCam")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks to select goal position."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_goal = (x, y)
            self.path_in_progress = True
            print(f"[GOAL] Selected goal at pixel position: ({x}, {y})")
    
    def video_capture_thread(self):
        """Background thread for capturing video frames."""
        print("[THREAD] Video capture thread started")
        frame_count = 0
        
        while not self.stop_flag:
            ret, frame = self.video_capture.read()
            
            if not ret:
                print("[THREAD] Failed to read frame, attempting to reconnect...")
                self.connect_camera()
                continue
            
            with self.frame_lock:
                self.current_frame = frame
            
            frame_count += 1
            if frame_count % 60 == 0:
                print(f"[THREAD] Video frames captured: {frame_count}")
        
        print("[THREAD] Video capture thread stopped")
    
    def find_car_marker(self, yolo_detections):
        """
        Find the pizza slice marker on the RC car (class 30 in COCO).
        
        Returns:
            (x, y, w, h) of pizza/car marker, or None if not found
        """
        for x, y, w, h, class_id, confidence in yolo_detections:
            if class_id == 30:  # Pizza class in COCO
                return (x, y, w, h)
        return None
    
    def draw_obstacles(self, frame, color_detections, yolo_detections):
        """Draw bounding boxes for detected obstacles."""
        display_frame = frame.copy()
        
        # Draw orange block detections (Jenga blocks)
        for x, y, w, h in color_detections:
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
            cv2.putText(display_frame, "Jenga Block", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        
        # Draw YOLO detections
        for x, y, w, h, class_id, confidence in yolo_detections:
            color = self.yolo_detector.get_class_color(class_id)
            color = tuple(map(int, color))
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            
            class_name = self.yolo_detector.get_class_name(class_id)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(display_frame, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return display_frame
    
    def draw_ui_overlay(self, frame):
        """Draw UI information overlay on the frame."""
        height, width = frame.shape[:2]
        overlay = frame.copy()
        
        # Draw semi-transparent status bar
        cv2.rectangle(overlay, (0, 0), (width, 140), (0, 0, 0), -1)
        alpha = 0.7
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Status
        status_text = "● RUNNING" if self.is_running else "○ STOPPED"
        status_color = (0, 255, 0) if self.is_running else (0, 0, 255)
        cv2.putText(frame, status_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.2, status_color, 2)
        
        # Car position
        if self.car_position:
            car_text = f"Car: ({self.car_position[0]}, {self.car_position[1]})"
            cv2.putText(frame, car_text, (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 0, 255), 1)
        else:
            cv2.putText(frame, "Car: NOT DETECTED", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 0, 255), 2)
        
        # Goal
        if self.selected_goal:
            goal_text = f"Goal: ({self.selected_goal[0]}, {self.selected_goal[1]})"
            cv2.putText(frame, goal_text, (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 0), 1)
        else:
            cv2.putText(frame, "Goal: Click screen to set", (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (100, 100, 255), 1)
        
        # Instructions
        instructions = "SPACE: Start/Stop | CLICK: Set Goal | R: Reset | Q: Quit"
        cv2.putText(frame, instructions, (15, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (200, 200, 200), 1)
        
        # Goal marker
        if self.selected_goal:
            cv2.circle(frame, self.selected_goal, 12, (0, 255, 255), 3)
            cv2.putText(frame, "TARGET", (self.selected_goal[0] - 35, self.selected_goal[1] - 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return frame
    
    def process_frame(self, frame):
        """Process a single frame for detections and pathfinding."""
        frame_height, frame_width = frame.shape[:2]
        
        # Detect orange blocks (Jenga blocks as barriers)
        color_detections, mask = self.color_detector.detect(frame)
        
        # Detect objects with YOLO (pizza marker for car position)
        yolo_detections = self.yolo_detector.detect(frame)
        
        # Find car position from pizza marker
        car_marker = self.find_car_marker(yolo_detections)
        
        if car_marker:
            car_x, car_y, car_w, car_h = car_marker
            self.car_position = (car_x + car_w // 2, car_y + car_h // 2)
        else:
            self.car_position = None
        
        # Build grid map from detections
        self.pathfinder.build_grid(frame, color_detections, yolo_detections)
        
        return {
            'color_detections': color_detections,
            'yolo_detections': yolo_detections,
            'car_marker': car_marker,
            'mask': mask,
            'height': frame_height,
            'width': frame_width
        }
    
    def calculate_navigation_command(self, frame_width, frame_height):
        """Calculate the next navigation command based on goal and obstacles."""
        if not self.is_running or not self.selected_goal or not self.car_position:
            return "E"
        
        # Convert pixel coordinates to grid coordinates
        car_grid_x = int(self.car_position[0] * (self.pathfinder.grid_width / frame_width))
        car_grid_y = int(self.car_position[1] * (self.pathfinder.grid_height / frame_height))
        
        goal_pixel_x, goal_pixel_y = self.selected_goal
        goal_grid_x = int(goal_pixel_x * (self.pathfinder.grid_width / frame_width))
        goal_grid_y = int(goal_pixel_y * (self.pathfinder.grid_height / frame_height))
        
        # Check if we've reached the goal
        distance_to_goal = np.sqrt(
            (self.car_position[0] - goal_pixel_x)**2 + 
            (self.car_position[1] - goal_pixel_y)**2
        )
        
        if distance_to_goal < self.arrival_threshold:
            print(f"[NAVIGATION] ✓ Goal reached!")
            self.selected_goal = None
            self.path_in_progress = False
            return "E"
        
        # Find path to goal
        path = self.pathfinder.find_path(car_grid_x, car_grid_y, goal_grid_x, goal_grid_y)
        
        if not path or len(path) < 2:
            print(f"[NAVIGATION] ✗ No path to goal found! Obstacles blocking the way.")
            return "E"
        
        # Get next waypoint
        waypoint = self.pathfinder.get_next_waypoint(car_grid_x, car_grid_y)
        
        if waypoint:
            command = self.pathfinder.get_direction_to_waypoint(
                car_grid_x, car_grid_y, waypoint[0], waypoint[1]
            )
            return command
        
        return "E"
    
    def run(self):
        """Main execution loop for the interactive navigator."""
        try:
            self.setup_detectors()
            self.connect_camera()
            
            print("\n" + "="*60)
            print("RC CAR INTERACTIVE NAVIGATOR")
            print("="*60)
            print("Controls:")
            print("  SPACE  - Start/Stop navigation")
            print("  CLICK  - Click screen to set destination")
            print("  R      - Reset navigation (clear goal)")
            print("  Q      - Quit")
            print("="*60 + "\n")
            
            # Start video capture thread
            self.video_thread = threading.Thread(target=self.video_capture_thread, daemon=True)
            self.video_thread.start()
            
            # Wait for first frame
            time.sleep(1)
            
            # Create window and set mouse callback
            window_name = "RC Car Live Navigator"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1280, 720)
            cv2.setMouseCallback(window_name, self.mouse_callback)
            
            frame_count = 0
            command_count = 0
            last_command_time = time.time()
            command_interval = 0.1  # Send commands every 100ms
            
            print("[MAIN] Starting main loop...\n")
            
            while True:
                # Get current frame
                with self.frame_lock:
                    if self.current_frame is None:
                        time.sleep(0.01)
                        continue
                    frame = self.current_frame.copy()
                
                # Process frame for detections
                detection_data = self.process_frame(frame)
                
                # Calculate navigation command
                current_time = time.time()
                if current_time - last_command_time >= command_interval:
                    command = self.calculate_navigation_command(
                        detection_data['width'], 
                        detection_data['height']
                    )
                    
                    # Send command to car
                    if self.is_running or self.last_command != "E":
                        send_udp_command(self.car_ip, self.car_port, command)
                        self.last_command = command
                        command_count += 1
                    
                    last_command_time = current_time
                
                # Draw detections
                display_frame = self.draw_obstacles(
                    frame, 
                    detection_data['color_detections'],
                    detection_data['yolo_detections']
                )
                
                # Draw grid visualization
                display_frame = self.pathfinder.visualize_grid(display_frame)
                
                # Draw UI overlay
                display_frame = self.draw_ui_overlay(display_frame)
                
                # Display frames
                cv2.imshow(window_name, display_frame)
                cv2.imshow("Mask (Orange Blocks)", detection_data['mask'])
                
                frame_count += 1
                
                # Print status every 60 frames (roughly every 2 seconds)
                if frame_count % 60 == 0:
                    print(f"[MAIN] Frames: {frame_count} | Commands sent: {command_count} | "
                          f"Status: {'RUNNING' if self.is_running else 'STOPPED'}")
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[MAIN] Quitting...")
                    break
                elif key == ord(' '):  # Spacebar
                    self.is_running = not self.is_running
                    status = "STARTED" if self.is_running else "STOPPED"
                    print(f"[MAIN] Car {status}")
                elif key == ord('r'):  # Reset
                    self.selected_goal = None
                    self.path_in_progress = False
                    print(f"[MAIN] Navigation reset")
            
            # Cleanup
            self.stop_flag = True
            if self.video_thread:
                self.video_thread.join(timeout=2)
            
            cv2.destroyAllWindows()
            if self.video_capture:
                self.video_capture.release()
            
            # Send final stop command
            send_udp_command(self.car_ip, self.car_port, "E")
            print("[MAIN] ✓ Shutdown complete\n")
            
        except Exception as e:
            print(f"\n[ERROR] {e}")
            self.stop_flag = True
            cv2.destroyAllWindows()
            if self.video_capture:
                self.video_capture.release()
            raise


def main():
    """Entry point for the interactive navigator."""
    # Configuration - UPDATE THESE WITH YOUR ACTUAL IPs
    DROIDCAM_IP = "10.34.13.69"      # IP of device running DroidCam
    DROIDCAM_PORT = 4747              # DroidCam port (default)
    CAR_IP = "10.42.1.222"            # IP of RC car
    CAR_PORT = 8888                   # Port the RC car listens on
    
    navigator = InteractiveNavigator(DROIDCAM_IP, DROIDCAM_PORT, CAR_IP, CAR_PORT)
    navigator.run()


if __name__ == "__main__":
    main()
