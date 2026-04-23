import cv2
import numpy as np
import os

class YOLODetector:
    """
    Detects objects using YOLOv4 model.
    """
    
    def __init__(self, weights_path="yolov4.weights", config_path="yolov4.cfg", 
                 names_path="coco.names", confidence_threshold=0.5, nms_threshold=0.4):
        """
        Initialize the YOLO detector.
        
        Args:
            weights_path: Path to YOLOv4 weights file
            config_path: Path to YOLOv4 config file
            names_path: Path to COCO names file
            confidence_threshold: Minimum confidence for detections
            nms_threshold: Non-maximum suppression threshold
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.net = None
        self.output_layers = None
        self.classes = []
        self.colors = None
        
        # Load model and class names
        self._load_model(weights_path, config_path)
        self._load_class_names(names_path)
    
    def _load_model(self, weights_path, config_path):
        """Load the YOLOv4 model"""
        # Check if files exist
        if not os.path.exists(weights_path):
            print(f"Error: {weights_path} not found. Please download YOLOv4 weights.")
            return
        if not os.path.exists(config_path):
            print(f"Error: {config_path} not found. Please download YOLOv4 config.")
            return
        
        try:
            self.net = cv2.dnn.readNet(weights_path, config_path)
            
            # Get layer names
            layer_names = self.net.getLayerNames()
            
            # Handle both old and new OpenCV versions
            unconnected_layers = self.net.getUnconnectedOutLayers()
            if isinstance(unconnected_layers[0], np.integer):
                self.output_layers = [layer_names[i - 1] for i in unconnected_layers]
            else:
                self.output_layers = unconnected_layers
            
            print("YOLO model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.net = None
    
    def _load_class_names(self, names_path):
        """Load COCO class names"""
        try:
            with open(names_path, "r") as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            # Generate random colors for each class
            self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
            print(f"Loaded {len(self.classes)} class names")
        except FileNotFoundError:
            print(f"Warning: {names_path} not found. Using default class names.")
            self.classes = [f"class_{i}" for i in range(80)]
            self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
    
    def is_loaded(self):
        """Check if model is loaded successfully"""
        return self.net is not None
    
    def detect(self, frame):
        """
        Detect objects in a frame using YOLO.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            detections: List of (x, y, w, h, class_id, confidence) tuples
        """
        if not self.is_loaded() or frame is None:
            return []
        
        try:
            height, width = frame.shape[:2]
            
            # Create blob from frame
            blob = cv2.dnn.blobFromImage(
                frame,
                scalefactor=0.00392,
                size=(416, 416),
                mean=(0, 0, 0),
                swapRB=True,
                crop=False
            )
            
            # Set input and perform forward pass
            self.net.setInput(blob)
            outputs = self.net.forward(self.output_layers)
            
            boxes, confidences, class_ids = [], [], []
            
            # Process each output
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    # Only keep detections with sufficient confidence
                    if confidence > self.confidence_threshold:
                        # YOLO output: center_x, center_y, width, height (normalized)
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        # Convert to top-left corner coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        # Clamp coordinates to frame bounds
                        x = max(0, min(x, width - 1))
                        y = max(0, min(y, height - 1))
                        w = max(1, w)
                        h = max(1, h)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # Apply Non-Maximum Suppression
            detections = []
            if boxes:
                indices = cv2.dnn.NMSBoxes(
                    boxes,
                    confidences,
                    score_threshold=self.confidence_threshold,
                    nms_threshold=self.nms_threshold
                )
                
                detections = [
                    (boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], class_ids[i], confidences[i])
                    for i in indices.flatten()
                ]
            
            return detections
        
        except Exception as e:
            print(f"Error during YOLO detection: {e}")
            return []
    
    def get_class_name(self, class_id):
        """Get class name by ID"""
        if 0 <= class_id < len(self.classes):
            return self.classes[class_id]
        return f"unknown_{class_id}"
    
    def get_class_color(self, class_id):
        """Get color for a class ID"""
        if self.colors is not None and class_id < len(self.colors):
            return self.colors[class_id]
        return (0, 255, 0)