import cv2
import numpy as np
from color_detector import OrangeBlockDetector
from block_distance_analyzer import JengaBlockAnalyzer

class GridMapper:
    """
    Converts top-down camera feed into a binary grid matrix with real-world calibration.
    Uses Jenga block analyzer for accurate distance estimation.
    
    Real-world calibration:
    - Each grid cell represents 20cm x 20cm in the real world
    - RC car dimensions: 16cm x 21cm
    - Uses lens distortion correction for accurate obstacle detection
    """
    
    def __init__(self, grid_width=40, grid_height=40, frame_width=640, frame_height=480,
                 cell_size_cm=20, car_width_cm=16, car_height_cm=21, car_safety_radius_cm=15):
        """
        Initialize the grid mapper with Jenga block analyzer.
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Real-world calibration
        self.cell_size_cm = cell_size_cm
        self.car_width_cm = car_width_cm
        self.car_height_cm = car_height_cm
        self.car_safety_radius_cm = car_safety_radius_cm
        
        # Calculate cell size in pixels
        self.cell_width_px = frame_width / grid_width
        self.cell_height_px = frame_height / grid_height
        
        # Initialize grid
        self.grid = np.zeros((grid_height, grid_width), dtype=np.uint8)
        
        # Initialize block analyzer for accurate distance/sizing
        self.block_analyzer = JengaBlockAnalyzer()
        
        # Color detector (fallback for basic detection)
        self.color_detector = OrangeBlockDetector(
            lower_h=0, lower_s=210, lower_v=100,
            upper_h=25, upper_s=255, upper_v=255,
            min_area=100
        )
        
        # Print calibration info
        print(f"\n{'='*60}")
        print("GRID MAPPER - WITH BLOCK ANALYSIS")
        print(f"{'='*60}")
        print(f"Grid size: {grid_width}x{grid_height} cells")
        print(f"Grid cell size: {cell_size_cm}cm x {cell_size_cm}cm")
        print(f"Total maze area: {grid_width * cell_size_cm}cm x {grid_height * cell_size_cm}cm")
        print(f"                 ({grid_width * cell_size_cm / 100}m x {grid_height * cell_size_cm / 100}m)")
        print(f"\nRC Car dimensions: {car_width_cm}cm x {car_height_cm}cm")
        print(f"Car safety radius: {car_safety_radius_cm}cm")
        print(f"\nCell size in pixels: {self.cell_width_px:.2f}px x {self.cell_height_px:.2f}px")
        print(f"{'='*60}\n")
    
    def add_calibration_point(self, frame, distance_cm):
        """
        Add a calibration point for the block analyzer.
        
        Args:
            frame: Frame with Jenga block at known distance
            distance_cm: Real distance to the block
        """
        return self.block_analyzer.add_calibration_point(frame, distance_cm)
    
    def pixel_to_grid(self, pixel_x, pixel_y):
        """
        Convert pixel coordinates to grid cell coordinates.
        """
        grid_x = int(pixel_x / self.cell_width_px)
        grid_y = int(pixel_y / self.cell_height_px)
        
        grid_x = max(0, min(grid_x, self.grid_width - 1))
        grid_y = max(0, min(grid_y, self.grid_height - 1))
        
        return grid_x, grid_y
    
    def grid_to_pixel(self, grid_x, grid_y):
        """
        Convert grid cell coordinates to pixel coordinates (center of cell).
        """
        pixel_x = int((grid_x + 0.5) * self.cell_width_px)
        pixel_y = int((grid_y + 0.5) * self.cell_height_px)
        
        return pixel_x, pixel_y
    
    def grid_to_real_world(self, grid_x, grid_y):
        """
        Convert grid cell coordinates to real-world coordinates in cm.
        """
        real_x_cm = (grid_x + 0.5) * self.cell_size_cm
        real_y_cm = (grid_y + 0.5) * self.cell_size_cm
        
        return real_x_cm, real_y_cm
    
    def is_car_safe_at(self, grid_x, grid_y, include_buffer=True):
        """
        Check if the car can safely be placed at a grid cell.
        """
        if include_buffer:
            safety_cells = 1
            
            for dy in range(-safety_cells, safety_cells + 1):
                for dx in range(-safety_cells, safety_cells + 1):
                    check_x = grid_x + dx
                    check_y = grid_y + dy
                    
                    if 0 <= check_x < self.grid_width and 0 <= check_y < self.grid_height:
                        if self.grid[check_y, check_x] == 1:
                            return False
            return True
        else:
            if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                return self.grid[grid_y, grid_x] == 0
            return False
    
    def build_grid(self, frame):
        """
        Build grid using accurate block analysis.
        """
        # Analyze blocks for accurate positioning
        analyses, mask = self.block_analyzer.analyze_all_blocks(frame)
        
        # Reset grid
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)
        
        # Mark walls based on accurate analysis
        for analysis in analyses:
            x, y, w, h = analysis['x'], analysis['y'], analysis['w'], analysis['h']
            
            # Get grid coordinates
            grid_x1, grid_y1 = self.pixel_to_grid(x, y)
            grid_x2, grid_y2 = self.pixel_to_grid(x + w, y + h)
            
            # Fill grid cells with walls
            for gy in range(grid_y1, grid_y2 + 1):
                for gx in range(grid_x1, grid_x2 + 1):
                    if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
                        self.grid[gy, gx] = 1
        
        return self.grid, mask, analyses
    
    def visualize_grid(self, frame, analyses):
        """
        Draw grid overlay with block analysis data.
        """
        frame_display = frame.copy()
        height, width = frame.shape[:2]
        
        # Draw grid lines
        for x in range(self.grid_width + 1):
            pixel_x = int(x * self.cell_width_px)
            cv2.line(frame_display, (pixel_x, 0), (pixel_x, height), (100, 100, 100), 1)
        
        for y in range(self.grid_height + 1):
            pixel_y = int(y * self.cell_height_px)
            cv2.line(frame_display, (0, pixel_y), (width, pixel_y), (100, 100, 100), 1)
        
        # Draw wall cells
        for gy in range(self.grid_height):
            for gx in range(self.grid_width):
                if self.grid[gy, gx] == 1:
                    pixel_x, pixel_y = self.grid_to_pixel(gx, gy)
                    top_left = (int(pixel_x - self.cell_width_px/2), int(pixel_y - self.cell_height_px/2))
                    bottom_right = (int(pixel_x + self.cell_width_px/2), int(pixel_y + self.cell_height_px/2))
                    cv2.rectangle(frame_display, top_left, bottom_right, (0, 0, 255), -1)
        
        # Draw block analysis boxes with distance info
        for analysis in analyses:
            x, y, w, h = analysis['x'], analysis['y'], analysis['w'], analysis['h']
            distance = analysis['estimated_distance_cm']
            
            # Draw bounding box
            color = (0, 255, 0) if analysis['has_correction'] else (0, 165, 255)
            cv2.rectangle(frame_display, (x, y), (x+w, y+h), color, 2)
            
            # Draw distance
            if distance:
                distance_text = f"{distance:.1f}cm"
                cv2.putText(frame_display, distance_text, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return frame_display
    
    def print_grid(self):
        """Print the grid matrix to console."""
        print("\nGrid Matrix (1=wall, 0=empty):")
        print("=" * (self.grid_width + 2))
        for row in self.grid:
            print("".join([str(cell) for cell in row]))
        print("=" * (self.grid_width + 2))
    
    def print_grid_stats(self):
        """Print grid statistics."""
        wall_count = np.sum(self.grid)
        empty_count = self.grid.size - wall_count
        wall_percent = (wall_count / self.grid.size) * 100
        
        print(f"\nGrid Statistics:")
        print(f"  Total cells: {self.grid.size}")
        print(f"  Wall cells: {wall_count} ({wall_percent:.1f}%)")
        print(f"  Empty cells: {empty_count} ({100-wall_percent:.1f}%)")
        print(f"  Maze area: {self.grid_width * self.cell_size_cm}cm × {self.grid_height * self.cell_size_cm}cm")
        print(f"  Calibration points: {len(self.block_analyzer.calibration_points)}")
        print(f"  Distortion correction: {'YES' if self.block_analyzer.distortion_correction is not None else 'NO'}")
    
    def save_grid(self, filename):
        """Save grid matrix to file."""
        np.savetxt(filename, self.grid, fmt='%d', delimiter='')
        print(f"Grid saved to {filename}")
    
    def get_grid(self):
        """Get the current grid matrix."""
        return self.grid.copy()


def test_grid_mapper():
    """Interactive grid mapper test with block analysis."""
    print("\n" + "="*60)
    print("GRID MAPPER TEST WITH BLOCK ANALYSIS")
    print("="*60)
    print("\nInitializing grid mapper...")
    
    grid_mapper = GridMapper(
        grid_width=40,
        grid_height=40,
        cell_size_cm=20,
        car_width_cm=16,
        car_height_cm=21,
        car_safety_radius_cm=15
    )
    
    # DroidCam settings
    droidcam_ip = "10.31.0.129"
    droidcam_port = 4747
    
    print(f"Connecting to DroidCam at {droidcam_ip}:{droidcam_port}...")
    cap = cv2.VideoCapture(f"http://{droidcam_ip}:{droidcam_port}/video")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("Failed to connect to DroidCam. Check IP and port.")
        return
    
    print("✓ Connected to DroidCam!")
    print("\nCalibration Instructions:")
    print("  1. Position block at 14cm, press 1")
    print("  2. Position block at 24cm, press 2")
    print("  3. Position block at 42cm, press 3")
    print("  4. Position block at 80cm, press 4")
    print("  5. Press SPACE to build grid with calibration\n")
    print("Controls:")
    print("  1/2/3/4 - Add calibration point")
    print("  SPACE   - Build and print grid")
    print("  P       - Print grid matrix")
    print("  T       - Print statistics")
    print("  S       - Save grid to file")
    print("  R       - Reset calibration")
    print("  Q       - Quit\n")
    
    frame_count = 0
    current_analyses = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to get frame.")
            continue
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        if grid_mapper.frame_width != width or grid_mapper.frame_height != height:
            grid_mapper.frame_width = width
            grid_mapper.frame_height = height
            grid_mapper.cell_width_px = width / grid_mapper.grid_width
            grid_mapper.cell_height_px = height / grid_mapper.grid_height
        
        # Build grid with analysis
        grid, mask, analyses = grid_mapper.build_grid(frame)
        current_analyses = analyses
        
        # Visualize
        frame_display = grid_mapper.visualize_grid(frame, analyses)
        
        # Add info panel
        cv2.rectangle(frame_display, (0, 0), (width, 120), (0, 0, 0), -1)
        cv2.putText(frame_display, "GRID MAPPER - WITH BLOCK ANALYSIS", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame_display, f"Blocks: {len(analyses)} | Cal points: {len(grid_mapper.block_analyzer.calibration_points)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 165), 1)
        cv2.putText(frame_display, f"Correction: {'YES' if grid_mapper.block_analyzer.distortion_correction is not None else 'NO'}", 
                   (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 165), 1)
        cv2.putText(frame_display, "1/2/3/4:Cal | SPACE:Build | P:Print | T:Stats | S:Save | R:Reset | Q:Quit",
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Display wall count
        wall_count = np.sum(grid)
        empty_count = grid.size - wall_count
        cv2.putText(frame_display, f"Walls: {wall_count} | Empty: {empty_count}",
                   (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 165), 1)
        
        cv2.imshow("Grid Mapper", frame_display)
        cv2.imshow("Orange Block Mask", mask)
        
        frame_count += 1
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            grid_mapper.add_calibration_point(frame, 14)
        elif key == ord('2'):
            grid_mapper.add_calibration_point(frame, 24)
        elif key == ord('3'):
            grid_mapper.add_calibration_point(frame, 42)
        elif key == ord('4'):
            grid_mapper.add_calibration_point(frame, 80)
        elif key == ord('r'):
            grid_mapper.block_analyzer.calibration_points = []
            grid_mapper.block_analyzer.distortion_correction = None
            print("\n✓ Calibration reset")
        elif key == ord('p'):
            grid_mapper.print_grid()
        elif key == ord('t'):
            grid_mapper.print_grid_stats()
        elif key == ord('s'):
            grid_mapper.save_grid("maze_grid.txt")
        elif key == ord(' '):
            print(f"\n{'='*60}")
            print(f"Grid built from frame {frame_count}")
            print(f"{'='*60}")
            grid_mapper.print_grid()
            grid_mapper.print_grid_stats()
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Grid mapper test complete!")


if __name__ == "__main__":
    test_grid_mapper()