import cv2
import numpy as np
from color_detector import OrangeBlockDetector
import math

class JengaBlockAnalyzer:
    """
    Analyzes Jenga blocks with lens distortion correction.
    Learns correction curve from multiple calibration points.
    """
    
    # Jenga block real dimensions (in cm)
    LONG_SIDE = 7.5
    SHORT_SIDE = 3.5
    HEIGHT = 1.5
    
    def __init__(self, camera_focal_length=800):
        """
        Initialize the Jenga block analyzer.
        """
        self.focal_length = camera_focal_length
        self.calibration_points = []  # [(actual_distance, pixel_width), ...]
        self.distortion_correction = None  # Will be a polynomial
        
        # Orange block detector
        self.color_detector = OrangeBlockDetector(
            lower_h=5, lower_s=200, lower_v=100,
            upper_h=25, upper_s=255, upper_v=255,
            min_area=100
        )
    
    def estimate_edge_type(self, pixel_width, pixel_height):
        """
        Determine if viewing long side (7.5cm) or short side (3.5cm).
        """
        aspect_ratio = pixel_width / pixel_height if pixel_height > 0 else 0
        
        if aspect_ratio > 2.5:
            confidence = min(1.0, (aspect_ratio - 2.5) / 2.0)
            return 'long', confidence
        elif aspect_ratio > 1.5:
            confidence = (aspect_ratio - 1.5) / 1.0
            return 'long', confidence
        elif aspect_ratio > 0.8:
            confidence = 0.5
            return 'unknown', confidence
        else:
            return 'short', 0.7
    
    def calculate_distance_from_size(self, pixel_size, real_size_cm, focal_length=None):
        """
        Calculate distance using pinhole camera model.
        """
        if focal_length is None:
            focal_length = self.focal_length
        
        if pixel_size <= 0:
            return None
        
        distance_cm = (real_size_cm * focal_length) / pixel_size
        return distance_cm
    
    def add_calibration_point(self, frame, actual_distance_cm):
        """
        Add a calibration point at a known distance.
        
        Args:
            frame: Frame with Jenga block at known distance
            actual_distance_cm: Real distance to the block
        """
        detections, _ = self.color_detector.detect(frame)
        
        if detections:
            x, y, w, h = detections[0]
            self.calibration_points.append((actual_distance_cm, w))
            
            print(f"\n✓ Calibration point added:")
            print(f"  Actual distance: {actual_distance_cm}cm")
            print(f"  Pixel width: {w}px")
            print(f"  Total calibration points: {len(self.calibration_points)}")
            
            # If we have multiple points, build correction curve
            if len(self.calibration_points) >= 2:
                self.build_distortion_correction()
            
            return True
        else:
            print("No blocks detected for calibration")
            return False
    
    def build_distortion_correction(self):
        """
        Build a polynomial correction curve from calibration points.
        Uses least-squares fitting to find the best correction.
        """
        if len(self.calibration_points) < 2:
            print("Need at least 2 calibration points")
            return
        
        # Extract actual distances and pixel widths
        actual_distances = np.array([p[0] for p in self.calibration_points])
        pixel_widths = np.array([p[1] for p in self.calibration_points])
        
        # Calculate what focal length would be for each point
        focal_lengths = (pixel_widths * actual_distances) / self.LONG_SIDE
        
        # Fit polynomial to focal_length vs distance
        # This captures how lens distortion changes with distance
        degree = min(2, len(self.calibration_points) - 1)  # Use degree 2 or less
        self.distortion_correction = np.polyfit(actual_distances, focal_lengths, degree)
        
        print(f"\n{'='*60}")
        print("DISTORTION CORRECTION CURVE BUILT")
        print(f"{'='*60}")
        print(f"Calibration points used: {len(self.calibration_points)}")
        print(f"Polynomial degree: {degree}")
        print(f"Coefficients: {self.distortion_correction}")
        
        # Print the correction function
        print(f"\nDistance-dependent focal length:")
        for i, actual_dist in enumerate([10, 20, 30, 40, 50]):
            corrected_fl = np.polyval(self.distortion_correction, actual_dist)
            print(f"  At {actual_dist}cm: focal_length = {corrected_fl:.1f}px")
        
        print(f"{'='*60}\n")
    
    def get_corrected_focal_length(self, estimated_distance_cm):
        """
        Get the distance-dependent focal length correction.
        
        Args:
            estimated_distance_cm: Initial distance estimate
            
        Returns:
            float: Corrected focal length
        """
        if self.distortion_correction is None:
            return self.focal_length
        
        # Apply polynomial correction
        corrected_fl = np.polyval(self.distortion_correction, estimated_distance_cm)
        return corrected_fl
    
    def calculate_distance_with_correction(self, pixel_size, real_size_cm):
        """
        Calculate distance with lens distortion correction.
        Iteratively refines the estimate.
        
        Args:
            pixel_size: Size in pixels
            real_size_cm: Known real size in cm
            
        Returns:
            float: Corrected distance estimate
        """
        # Start with basic estimate
        distance = self.calculate_distance_from_size(pixel_size, real_size_cm, self.focal_length)
        
        if self.distortion_correction is None or distance is None:
            return distance
        
        # Iteratively refine (usually converges in 2-3 iterations)
        for iteration in range(5):
            # Get corrected focal length for this distance
            corrected_fl = self.get_corrected_focal_length(distance)
            
            # Recalculate distance with corrected focal length
            new_distance = self.calculate_distance_from_size(pixel_size, real_size_cm, corrected_fl)
            
            # Check if converged
            if abs(new_distance - distance) < 0.1:  # Less than 0.1cm difference
                break
            
            distance = new_distance
        
        return distance
    
    def analyze_block(self, frame, detection):
        """
        Complete analysis of a single Jenga block with correction.
        """
        x, y, w, h = detection
        aspect_ratio = w / h if h > 0 else 0
        
        # Determine edge type
        edge_type, confidence = self.estimate_edge_type(w, h)
        
        # Calculate distance
        distance = None
        if edge_type == 'long':
            if self.distortion_correction is not None:
                distance = self.calculate_distance_with_correction(w, self.LONG_SIDE)
            else:
                distance = self.calculate_distance_from_size(w, self.LONG_SIDE)
        elif edge_type == 'short':
            if self.distortion_correction is not None:
                distance = self.calculate_distance_with_correction(w, self.SHORT_SIDE)
            else:
                distance = self.calculate_distance_from_size(w, self.SHORT_SIDE)
        else:
            if self.distortion_correction is not None:
                distance = self.calculate_distance_with_correction(w, self.LONG_SIDE)
            else:
                distance = self.calculate_distance_from_size(w, self.LONG_SIDE)
        
        analysis = {
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'aspect_ratio': aspect_ratio,
            'edge_type': edge_type,
            'confidence': confidence,
            'pixel_width': w,
            'pixel_height': h,
            'estimated_distance_cm': distance,
            'long_side': self.LONG_SIDE,
            'short_side': self.SHORT_SIDE,
            'has_correction': self.distortion_correction is not None
        }
        
        return analysis
    
    def analyze_all_blocks(self, frame):
        """Analyze all detected Jenga blocks in a frame."""
        detections, mask = self.color_detector.detect(frame)
        
        analyses = []
        for detection in detections:
            analysis = self.analyze_block(frame, detection)
            analyses.append(analysis)
        
        return analyses, mask
    
    def draw_analysis(self, frame, analyses):
        """Draw analysis results on frame."""
        frame_display = frame.copy()
        
        for i, analysis in enumerate(analyses):
            x, y, w, h = analysis['x'], analysis['y'], analysis['w'], analysis['h']
            edge_type = analysis['edge_type']
            distance = analysis['estimated_distance_cm']
            aspect_ratio = analysis['aspect_ratio']
            confidence = analysis['confidence']
            has_correction = analysis['has_correction']
            
            # Color based on edge type
            if edge_type == 'long':
                color = (0, 255, 0)
            elif edge_type == 'short':
                color = (255, 0, 0)
            else:
                color = (0, 165, 255)
            
            # Thicker border if corrected
            thickness = 4 if has_correction else 3
            cv2.rectangle(frame_display, (x, y), (x+w, y+h), color, thickness)
            
            # Edge type label
            if edge_type == 'long':
                size_label = f"LONG (7.5cm) - {w}px"
                size_color = (0, 255, 0)
            elif edge_type == 'short':
                size_label = f"SHORT (3.5cm) - {w}px"
                size_color = (255, 0, 0)
            else:
                size_label = f"AMBIGUOUS - {w}px"
                size_color = (0, 165, 255)
            
            cv2.putText(frame_display, size_label, (x, y - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, size_color, 2)
            cv2.putText(frame_display, f"Confidence: {confidence:.1%}", (x, y - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, size_color, 1)
            
            # Distance with correction indicator
            if distance is not None:
                if has_correction:
                    distance_label = f"~{distance:.1f}cm (corrected)"
                    distance_color = (0, 255, 255)
                else:
                    distance_label = f"~{distance:.1f}cm"
                    distance_color = (255, 255, 0)
                
                cv2.putText(frame_display, distance_label, (x, y + h + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, distance_color, 2)
            
            # Aspect ratio
            aspect_label = f"AR: {aspect_ratio:.2f}"
            cv2.putText(frame_display, aspect_label, (x, y + h + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame_display


def test_jenga_analyzer():
    """Interactive test for Jenga block analyzer with correction."""
    print("\n" + "="*60)
    print("JENGA BLOCK ANALYZER WITH DISTORTION CORRECTION")
    print("="*60)
    print("\nInitializing analyzer...")
    
    analyzer = JengaBlockAnalyzer()
    
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
    print("\nCalibration Instructions:")
    print("  1. Position block at 14cm, press 1")
    print("  2. Position block at 24cm, press 2")
    print("  3. Position block at 42cm, press 3")
    print("  4. Press SPACE to test with all calibration points\n")
    print("Controls:")
    print("  1/2/3 - Add calibration point at that distance")
    print("  SPACE - Print analysis")
    print("  R     - Reset calibration")
    print("  Q     - Quit\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to get frame.")
            continue
        
        # Analyze blocks
        analyses, mask = analyzer.analyze_all_blocks(frame)
        
        # Draw analysis
        frame_display = analyzer.draw_analysis(frame, analyses)
        
        # Add info panel
        height, width = frame_display.shape[:2]
        cv2.rectangle(frame_display, (0, 0), (width, 140), (0, 0, 0), -1)
        cv2.putText(frame_display, "JENGA ANALYZER - DISTORTION CORRECTION", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame_display, f"Blocks: {len(analyses)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 165), 1)
        cv2.putText(frame_display, f"Calibration points: {len(analyzer.calibration_points)}", (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 165), 1)
        cv2.putText(frame_display, f"Focal length: {analyzer.focal_length:.1f}px", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame_display, "1:Cal@14cm | 2:Cal@24cm | 3:Cal@42cm | SPACE:Test | R:Reset | Q:Quit",
                   (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow("Jenga Block Analyzer", frame_display)
        cv2.imshow("Orange Mask", mask)
        
        frame_count += 1
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            analyzer.add_calibration_point(frame, 14)
        elif key == ord('2'):
            analyzer.add_calibration_point(frame, 24)
        elif key == ord('3'):
            analyzer.add_calibration_point(frame, 42)
        elif key == ord('r'):
            analyzer.calibration_points = []
            analyzer.distortion_correction = None
            print("\n✓ Calibration reset")
        elif key == ord(' '):
            if analyses:
                print(f"\n{'='*60}")
                print(f"Frame {frame_count} Analysis:")
                print(f"{'='*60}")
                for i, analysis in enumerate(analyses):
                    print(f"\nBlock {i+1}:")
                    print(f"  Pixel width: {analysis['pixel_width']}px")
                    print(f"  Aspect ratio: {analysis['aspect_ratio']:.2f}")
                    print(f"  Edge type: {analysis['edge_type']}")
                    print(f"  Confidence: {analysis['confidence']:.1%}")
                    if analysis['estimated_distance_cm']:
                        corr = " (with correction)" if analysis['has_correction'] else ""
                        print(f"  Estimated distance: {analysis['estimated_distance_cm']:.1f}cm{corr}")
                    else:
                        print(f"  Estimated distance: Could not determine")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Test complete!")


if __name__ == "__main__":
    test_jenga_analyzer()