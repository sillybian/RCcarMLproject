import cv2
import numpy as np
import time

class DroidCamDetector:
    def __init__(self, ip, port, yolo_net, yolo_layer_names):
        """
        Initialize DroidCam detector.
        
        Args:
            ip: IP address of the device running DroidCam
            port: Port number (default 4747)
            yolo_net: YOLOv4 network object
            yolo_layer_names: Output layer names from YOLOv4
        """
        self.url = f"http://{ip}:{port}/video"
        self.cap = None
        self.net = yolo_net
        self.layer_names = yolo_layer_names
        self.retry_count = 0
        self.max_retries = 5
        
        # Try to connect to DroidCam
        self._connect()
    
    def _connect(self):
        """Establish connection to DroidCam video stream"""
        try:
            self.cap = cv2.VideoCapture(self.url)
            
            # Set connection timeout and buffer size
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Try to read first frame to verify connection
            ret, _ = self.cap.read()
            if ret:
                print(f"Successfully connected to DroidCam at {self.url}")
                self.retry_count = 0
            else:
                raise Exception("Could not read from video stream")
                
        except Exception as e:
            print(f"Error connecting to DroidCam: {e}")
            if self.cap:
                self.cap.release()
            self.cap = None

    def get_frame(self):
        """
        Get the next frame from DroidCam.
        
        Returns:
            frame: The captured frame, or None if failed
        """
        if self.cap is None or not self.cap.isOpened():
            print("Attempting to reconnect to DroidCam...")
            self._connect()
            if self.cap is None:
                return None
        
        try:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame, attempting reconnection...")
                self._connect()
                return None
            
            # Convert BGR to RGB if needed (DroidCam might send different format)
            if frame is not None:
                self.retry_count = 0
                return frame
            else:
                return None
                
        except Exception as e:
            print(f"Error reading frame: {e}")
            self._connect()
            return None

    def detect_objects(self, frame):
        """
        Detect objects in the frame using YOLOv4.
        
        Args:
            frame: Input frame from video
            
        Returns:
            List of detections in format [(x, y, w, h, class_id), ...]
        """
        if frame is None:
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
            outputs = self.net.forward(self.layer_names)
            
            boxes, confidences, class_ids = [], [], []
            confidence_threshold = 0.5
            
            # Process each output
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    # Only keep detections with sufficient confidence
                    if confidence > confidence_threshold:
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
            
            # Apply Non-Maximum Suppression to remove overlapping boxes
            if boxes:
                indices = cv2.dnn.NMSBoxes(
                    boxes, 
                    confidences, 
                    score_threshold=0.5, 
                    nms_threshold=0.4
                )
                
                # Return filtered detections
                return [(boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], class_ids[i]) 
                        for i in indices.flatten()]
            else:
                return []
                
        except Exception as e:
            print(f"Error during object detection: {e}")
            return []

    def release(self):
        """Release the video capture resource"""
        if self.cap is not None:
            self.cap.release()
            print("DroidCam connection closed")