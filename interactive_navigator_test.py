"""
Interactive Navigator Test - No RC Car

This is a test version of the interactive navigator that works without 
sending actual commands to the RC car. Perfect for testing the pathfinding,
object detection, and visualization before deploying to the real car.

Features:
- Live video feed from DroidCam
- Click to set navigation goals
- Shows path planning around orange blocks
- Displays car position (pizza marker)
- NO actual car movement commands sent
"""

import cv2
import numpy as np
from color_detector import OrangeBlockDetector
from yolo_detector import YOLODetector
from pathfinder import GridPathfinder
import threading
import time


class InteractiveNavigatorTest:
    def __init__(self, droidcam_ip, droidcam_port):
        """
        Initialize the test navigator (no RC car).
        
        Args:
            droidcam_ip: IP address of device running DroidCam
            droidcam_port: Port number for DroidCam (default 4747)
        """
        self.droidcam_ip = droidcam_ip
        self.droidcam_port = droidcam_port
        
        # State variables
        self.selected_goal = None
        self.is_running = False
        self.current_frame = None
        self.car_position = None
        self.current_path = []
        self.arrival_threshold = 10  # pixels
        
        # Detectors and pathfinder
        self.color_detector = None
        self.yolo_detector = None
        self.pathfinder = None
        self.video_capture = None
        
        # Threading
        self.video_thread = None
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
        """Find the pizza slice marker on the RC car (class 30 in COCO)."""
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
    
    def draw_path(self, frame, path, frame_width, frame_height):
        """Draw the navigation path on the frame."""
        if not path or len(path) < 2:
            return frame
        
        display = frame.copy()
        
        # Convert grid path coordinates back to pixel coordinates
        cell_width = frame_width / self.pathfinder.grid_width
        cell_height = frame_height / self.pathfinder.grid_height
        
        # Draw path line
        for i in range(len(path) - 1):
            x1 = int(path[i][0] * cell_width + cell_width / 2)
            y1 = int(path[i][1] * cell_height + cell_height / 2)
            x2 = int(path[i + 1][0] * cell_width + cell_width / 2)
            y2 = int(path[i + 1][1] * cell_height + cell_height / 2)
            cv2.line(display, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Draw waypoint circles
        for i, (grid_x, grid_y) in enumerate(path):
            x = int(grid_x * cell_width + cell_width / 2)
            y = int(grid_y * cell_height + cell_height / 2)
            if i == 0:
                cv2.circle(display, (x, y), 8, (0, 255, 0), -1)  # Start
            elif i == len(path) - 1:
                cv2.circle(display, (x, y), 8, (255, 0, 0), -1)  # End
            else:
                cv2.circle(display, (x, y), 5, (0, 200, 255), -1)  # Intermediate
        
        return display
    
    def draw_car(self, frame, car_position):
        """Draw the car marker on the frame."""
        if not car_position:
            return frame
        
        display = frame.copy()
        x, y = car_position
        
        # Draw car as a circle with orientation indicator
        cv2.circle(display, (x, y), 12, (255, 0, 255), 3)
        cv2.circle(display, (x, y), 6, (255, 0, 255), -1)
        
        # Draw direction indicator (line pointing forward)
        forward_dist = 20
        cv2.line(display, (x, y), (x, y - forward_dist), (255, 0, 255), 2)
        
        return display
    
    def draw_ui_overlay(self, frame):
        """Draw UI information overlay on the frame."""
        height, width = frame.shape[:2]
        overlay = frame.copy()
        
        # Draw semi-transparent status bar
        cv2.rectangle(overlay, (0, 0), (width, 140), (0, 0, 0), -1)
        alpha = 0.7
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Status
        status_text = "● TESTING (No car)" if self.is_running else "○ PAUSED (No car)"
        status_color = (0, 255, 0) if self.is_running else (0, 0, 255)
        cv2.putText(frame, status_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0, status_color, 2)
        
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
        instructions = "SPACE: Pause/Resume | CLICK: Set Goal | R: Reset | Q: Quit"
        cv2.putText(frame, instructions, (15, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (200, 200, 200), 1)
        
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
    
    def calculate_path(self, frame_width, frame_height):
        """Calculate the navigation path (visualization only)."""
        if not self.selected_goal or not self.car_position:
            self.current_path = []
            return
        
        # Convert pixel coordinates to grid coordinates
        car_grid_x = int(self.car_position[0] * (self.pathfinder.grid_width / frame_width))
        car_grid_y = int(self.car_position[1] * (self.pathfinder.grid_height / frame_height))
        
        goal_pixel_x, goal_pixel_y = self.selected_goal
        goal_grid_x = int(goal_pixel_x * (self.pathfinder.grid_width / frame_width))
        goal_grid_y = int(goal_pixel_y * (self.pathfinder.grid_height / frame_height))
        
        # Find path to goal
        path = self.pathfinder.find_path(car_grid_x, car_grid_y, goal_grid_x, goal_grid_y)
        
        if path and len(path) >= 2:
            self.current_path = path
            print(f"[PATHFINDER] Path found with {len(path)} waypoints")
        else:
            print(f"[PATHFINDER] ✗ No path to goal found! Obstacles blocking the way.")
            self.current_path = []
    
    def run(self):
        """Main execution loop for the test navigator."""
        try:
            self.setup_detectors()
            self.connect_camera()
            
            print("\n" + "="*60)
            print("RC CAR NAVIGATOR - TEST MODE (No Car)")
            print("="*60)
            print("This is a test version - no commands sent to RC car")
            print("Controls:")
            print("  SPACE  - Pause/Resume pathfinding")
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
            window_name = "RC Car Navigator - TEST MODE"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1280, 720)
            cv2.setMouseCallback(window_name, self.mouse_callback)
            
            frame_count = 0
            path_calc_count = 0
            last_calc_time = time.time()
            calc_interval = 0.5  # Recalculate path every 500ms
            
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
                
                # Calculate path periodically
                current_time = time.time()
                if self.is_running and current_time - last_calc_time >= calc_interval:
                    self.calculate_path(detection_data['width'], detection_data['height'])
                    path_calc_count += 1
                    last_calc_time = current_time
                
                # Draw detections
                display_frame = self.draw_obstacles(
                    frame, 
                    detection_data['color_detections'],
                    detection_data['yolo_detections']
                )
                
                # Draw path
                display_frame = self.draw_path(
                    display_frame, 
                    self.current_path,
                    detection_data['width'],
                    detection_data['height']
                )
                
                # Draw car
                display_frame = self.draw_car(display_frame, self.car_position)
                
                # Draw goal marker
                if self.selected_goal:
                    cv2.circle(display_frame, self.selected_goal, 12, (0, 255, 255), 3)
                    cv2.putText(display_frame, "TARGET", (self.selected_goal[0] - 35, self.selected_goal[1] - 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Draw UI overlay
                display_frame = self.draw_ui_overlay(display_frame)
                
                # Display frames
                cv2.imshow(window_name, display_frame)
                cv2.imshow("Mask (Orange Blocks)", detection_data['mask'])
                
                frame_count += 1
                
                # Print status every 60 frames
                if frame_count % 60 == 0:
                    print(f"[MAIN] Frames: {frame_count} | Paths calculated: {path_calc_count} | "
                          f"Status: {'RUNNING' if self.is_running else 'PAUSED'}")
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[MAIN] Quitting...")
                    break
                elif key == ord(' '):  # Spacebar
                    self.is_running = not self.is_running
                    status = "RESUMED" if self.is_running else "PAUSED"
                    print(f"[MAIN] Test {status}")
                elif key == ord('r'):  # Reset
                    self.selected_goal = None
                    self.current_path = []
                    print(f"[MAIN] Navigation reset")
            
            # Cleanup
            self.stop_flag = True
            if self.video_thread:
                self.video_thread.join(timeout=2)
            
            cv2.destroyAllWindows()
            if self.video_capture:
                self.video_capture.release()
            
            print("[MAIN] ✓ Test complete\n")
            
        except Exception as e:
            print(f"\n[ERROR] {e}")
            self.stop_flag = True
            cv2.destroyAllWindows()
            if self.video_capture:
                self.video_capture.release()
            raise


def main():
    """Entry point for the test navigator."""
    # Configuration - UPDATE THIS WITH YOUR DROIDCAM IP
    DROIDCAM_IP = "10.34.13.69"      # IP of device running DroidCam
    DROIDCAM_PORT = 4747              # DroidCam port (default)
    
    navigator = InteractiveNavigatorTest(DROIDCAM_IP, DROIDCAM_PORT)
    navigator.run()


if __name__ == "__main__":
    main()
