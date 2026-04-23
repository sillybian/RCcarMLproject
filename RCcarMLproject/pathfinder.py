import cv2
import numpy as np
from collections import deque

class GridPathfinder:
    """
    Creates a 2D grid map from detected obstacles and finds the optimal path for the RC car.
    """
    
    def __init__(self, grid_width=50, grid_height=50, car_radius=3, wall_padding=2):
        """
        Initialize the pathfinder.
        
        Args:
            grid_width: Number of cells horizontally
            grid_height: Number of cells vertically
            car_radius: Radius of the car in grid cells (for collision detection)
            wall_padding: Extra padding around walls to keep car away from obstacles
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.car_radius = car_radius
        self.wall_padding = wall_padding
        self.grid = np.zeros((grid_height, grid_width), dtype=np.uint8)
        self.path = []
    
    def build_grid(self, frame, color_detections, yolo_detections):
        """
        Build the grid map from detected obstacles.
        
        Args:
            frame: The video frame (used to get dimensions)
            color_detections: List of (x, y, w, h) for orange blocks
            yolo_detections: List of (x, y, w, h, class_id, confidence) for other objects
        """
        frame_height, frame_width = frame.shape[:2]
        
        # Reset grid
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)
        
        # Scaling factors to convert pixel coordinates to grid coordinates
        x_scale = self.grid_width / frame_width
        y_scale = self.grid_height / frame_height
        
        # Mark orange blocks as walls
        for x, y, w, h in color_detections:
            self._mark_obstacle(x, y, w, h, x_scale, y_scale)
        
        # Mark YOLO detections as walls (except pizza - class_30)
        for x, y, w, h, class_id, confidence in yolo_detections:
            # Skip pizza (class 30 in COCO) - we want to navigate TO it, not away
            if class_id != 30:
                self._mark_obstacle(x, y, w, h, x_scale, y_scale)
    
    def _mark_obstacle(self, x, y, w, h, x_scale, y_scale):
        """
        Mark an obstacle and its padding on the grid.
        
        Args:
            x, y, w, h: Bounding box of the obstacle in pixels
            x_scale, y_scale: Conversion factors from pixels to grid cells
        """
        # Convert pixel coordinates to grid coordinates
        grid_x1 = max(0, int((x - self.wall_padding) * x_scale))
        grid_y1 = max(0, int((y - self.wall_padding) * y_scale))
        grid_x2 = min(self.grid_width, int((x + w + self.wall_padding) * x_scale))
        grid_y2 = min(self.grid_height, int((y + h + self.wall_padding) * y_scale))
        
        # Mark all cells in this region as blocked
        self.grid[grid_y1:grid_y2, grid_x1:grid_x2] = 1
    
    def is_walkable(self, grid_x, grid_y):
        """
        Check if a grid cell is walkable (considering car radius).
        
        Args:
            grid_x, grid_y: Grid coordinates
            
        Returns:
            True if the cell and surrounding area (car_radius) is free
        """
        # Check if point is within grid bounds
        if grid_x < self.car_radius or grid_x >= self.grid_width - self.car_radius:
            return False
        if grid_y < self.car_radius or grid_y >= self.grid_height - self.car_radius:
            return False
        
        # Check all cells within car_radius
        for dx in range(-self.car_radius, self.car_radius + 1):
            for dy in range(-self.car_radius, self.car_radius + 1):
                if self.grid[grid_y + dy, grid_x + dx] == 1:
                    return False
        return True
    
    def find_path(self, start_x, start_y, goal_x, goal_y):
        """
        Find the shortest path using BFS (Breadth-First Search).
        
        Args:
            start_x, start_y: Starting position in grid coordinates
            goal_x, goal_y: Goal position in grid coordinates
            
        Returns:
            List of (grid_x, grid_y) tuples representing the path, or empty list if no path exists
        """
        # Clamp coordinates to valid range
        start_x = max(self.car_radius, min(start_x, self.grid_width - self.car_radius - 1))
        start_y = max(self.car_radius, min(start_y, self.grid_height - self.car_radius - 1))
        goal_x = max(self.car_radius, min(goal_x, self.grid_width - self.car_radius - 1))
        goal_y = max(self.car_radius, min(goal_y, self.grid_height - self.car_radius - 1))
        
        # Check if start and goal are walkable
        if not self.is_walkable(start_x, start_y) or not self.is_walkable(goal_x, goal_y):
            return []
        
        # BFS pathfinding
        queue = deque([(start_x, start_y, [(start_x, start_y)])])
        visited = set()
        visited.add((start_x, start_y))
        
        # 4-directional movement (up, down, left, right)
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        
        while queue:
            x, y, path = queue.popleft()
            
            # Check if we reached the goal
            if x == goal_x and y == goal_y:
                self.path = path
                return path
            
            # Explore neighbors
            for dx, dy in directions:
                next_x, next_y = x + dx, y + dy
                
                if (next_x, next_y) not in visited and self.is_walkable(next_x, next_y):
                    visited.add((next_x, next_y))
                    queue.append((next_x, next_y, path + [(next_x, next_y)]))
        
        # No path found
        self.path = []
        return []
    
    def get_next_waypoint(self, current_grid_x, current_grid_y):
        """
        Get the next waypoint on the current path.
        
        Args:
            current_grid_x, current_grid_y: Current position in grid coordinates
            
        Returns:
            (grid_x, grid_y) of next waypoint, or None if no path or path is complete
        """
        if not self.path or len(self.path) < 2:
            return None
        
        # Return the first waypoint after current position
        return self.path[1] if len(self.path) > 1 else self.path[0]
    
    def visualize_grid(self, frame):
        """
        Create a visualization of the grid map overlaid on the frame.
        
        Args:
            frame: The video frame
            
        Returns:
            Frame with grid overlay
        """
        frame_height, frame_width = frame.shape[:2]
        visualization = frame.copy()
        
        # Draw grid cells
        cell_width = frame_width / self.grid_width
        cell_height = frame_height / self.grid_height
        
        for grid_y in range(self.grid_height):
            for grid_x in range(self.grid_width):
                x = int(grid_x * cell_width)
                y = int(grid_y * cell_height)
                w = int(cell_width)
                h = int(cell_height)
                
                if self.grid[grid_y, grid_x] == 1:
                    # Wall cell - red
                    cv2.rectangle(visualization, (x, y), (x + w, y + h), (0, 0, 255), -1)
                
                # Draw grid lines
                cv2.rectangle(visualization, (x, y), (x + w, y + h), (100, 100, 100), 1)
        
        # Draw path if it exists
        if self.path:
            for i in range(len(self.path) - 1):
                x1 = int(self.path[i][0] * cell_width)
                y1 = int(self.path[i][1] * cell_height)
                x2 = int(self.path[i + 1][0] * cell_width)
                y2 = int(self.path[i + 1][1] * cell_height)
                cv2.line(visualization, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return visualization
    
    def get_direction_to_waypoint(self, car_x, car_y, waypoint_x, waypoint_y):
        """
        Determine which direction the car should move to reach a waypoint.
        
        Args:
            car_x, car_y: Current car position in grid coordinates
            waypoint_x, waypoint_y: Target waypoint in grid coordinates
            
        Returns:
            Command string: 'A' (forward), 'L' (left), 'R' (right), or 'E' (stop)
        """
        dx = waypoint_x - car_x
        dy = waypoint_y - car_y
        
        # Calculate the direction angle
        angle = np.arctan2(dy, dx)
        
        # Since the car is pointing "up" (negative y), we need to adjust
        # Assume car's forward direction is (0, -1) and right direction is (1, 0)
        
        # Normalize angle to -180 to 180 degrees
        angle_deg = np.degrees(angle)
        
        # If waypoint is very close, stop
        distance = np.sqrt(dx**2 + dy**2)
        if distance < 1:
            return 'E'
        
        # Simple direction determination
        # Adjust these thresholds based on your car's turning behavior
        if -45 <= angle_deg <= 45:
            return 'A'  # Go forward (waypoint is ahead)
        elif 45 < angle_deg < 135:
            return 'L'  # Turn left
        elif -135 < angle_deg < -45:
            return 'R'  # Turn right
        else:
            return 'E'  # Stop (waypoint is behind, shouldn't happen with good pathfinding)