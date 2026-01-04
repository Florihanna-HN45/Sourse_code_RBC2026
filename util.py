import pathlib
#
import cv2
import numpy as np
import tensorflow as tf
import torch
import time
import logging
from collections import deque
from typing import List, Tuple, Dict, Optional

#Setup Logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OptimizedCubeDetector:
    """
    Production-ready cube symbol detector optimized for RPi3
    
    Key Features:
    - Smart caching to reduce redundant computation
    - Adaptive preprocessing based on lighting
    - Multi-scale detection for robustness
    - TFLite quantized model for speed
    """
    
    def __init__(self, 
                 yolo_model_path: str,
                 tflite_model_path: str,
                 confidence_threshold: float = 0.7):
        """
        Initialize detector with optimized models
        """
        logger.info("🚀 Initializing Optimized Cube Detector...")

        #fix đường dẫn cho Windows
        import pathlib
        temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
        
        # STAGE 1: Load YOLOv5n
        logger.info("📦 Loading YOLOv5n for cube detection...")
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'custom',
                                         path=yolo_model_path, 
                                         force_reload=False,
                                         verbose=False)
        self.yolo_model.conf = confidence_threshold
        self.yolo_model.iou = 0.45
        self.yolo_model.max_det = 5
        self.yolo_model.cpu()
        self.yolo_model.eval()
        
        # STAGE 2: Load TFLite Model
        logger.info("🔬 Loading TFLite INT8 model...")
        self.interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # STAGE 3: Configuration
        self.classes = ['FAKE', 'REAL']
        self.img_size = 320
        self.cnn_size = 64
        
        # Performance monitoring
        self.fps_queue = deque(maxlen=30)
        self.timing_stats = {
            'yolo': deque(maxlen=100),
            'extract': deque(maxlen=100),
            'classify': deque(maxlen=100)
        }
        
        # Smart caching
        self.result_cache = {}
        self.cache_threshold = 20
        
        # CLAHE for low-light
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
        logger.info("✅ Initialization complete!")
        self._print_config()
    
    def _print_config(self):
        logger.info("\n" + "="*50)
        logger.info("CONFIGURATION SUMMARY")
        logger.info("="*50)
        logger.info(f"YOLOv5n: {self.img_size}x{self.img_size}")
        logger.info(f"TFLite: {self.cnn_size}x{self.cnn_size}")
        logger.info(f"Conf Threshold: {self.yolo_model.conf}")
        logger.info("="*50 + "\n")
    
    def detect_cubes(self, frame: np.ndarray) -> Tuple[List[Dict], float]:
        """YOLOv5n cube detection"""
        start = time.time()
        h, w = frame.shape[:2]
        frame_resized = cv2.resize(frame, (self.img_size, self.img_size))
        
        with torch.no_grad():
            results = self.yolo_model(frame_resized)
        
        yolo_time = (time.time() - start) * 1000
        self.timing_stats['yolo'].append(yolo_time)
        
        detections = results.xyxy[0].cpu().numpy()
        boxes = []
        
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            x1 = int(x1 * w / self.img_size)
            y1 = int(y1 * h / self.img_size)
            x2 = int(x2 * w / self.img_size)
            y2 = int(y2 * h / self.img_size)
            
            boxes.append({
                'bbox': [x1, y1, x2-x1, y2-y1],
                'confidence': float(conf),
                'class': int(cls),
                'center': [(x1+x2)//2, (y1+y2)//2]
            })
        
        return boxes, yolo_time
    
    def extract_symbol(self, roi: np.ndarray) -> Optional[np.ndarray]:
        """Extract symbol với adaptive preprocessing"""
        if roi.shape[0] < 20 or roi.shape[1] < 20:
            return None
        
        start = time.time()
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        # Adaptive threshold
        if brightness < 80:
            enhanced = self.clahe.apply(gray)
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            symbol = self._center_crop(roi)
        else:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) < 100:
                symbol = self._center_crop(roi)
            else:
                x, y, w, h = cv2.boundingRect(c)
                padding = 5
                x, y, w, h = max(0, x-padding), max(0, y-padding), \
                           min(roi.shape[1]-x, w+10), min(roi.shape[0]-y, h+10)
                symbol = roi[y:y+h, x:x+w]
        
        symbol_resized = cv2.resize(symbol, (self.cnn_size, self.cnn_size))
        self.timing_stats['extract'].append((time.time() - start) * 1000)
        return symbol_resized
    
    def _center_crop(self, roi: np.ndarray) -> np.ndarray:
        h, w = roi.shape[:2]
        size = min(h, w)
        cx, cy = w//2, h//2
        x1 = max(0, cx - size//2)
        y1 = max(0, cy - size//2)
        return roi[y1:y1+size, x1:x1+size]
    
    def classify_symbol(self, symbol_img: np.ndarray) -> Tuple[str, float]:
        """TFLite INT8 classification"""
        if symbol_img is None:
            return "UNKNOWN", 0.0
        
        start = time.time()
        img = cv2.cvtColor(symbol_img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=(0, -1))
        
        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        class_id = np.argmax(output[0])
        confidence = float(output[0][class_id])
        
        self.timing_stats['classify'].append((time.time() - start) * 1000)
        return self.classes[class_id], confidence
    
    def _check_cache(self, center: List[int]) -> Optional[Dict]:
        """Smart cache check"""
        center_tuple = tuple(center)
        for cached_center, result in self.result_cache.items():
            distance = np.sqrt(sum((a-b)**2 for a, b in zip(center, cached_center)))
            if distance < self.cache_threshold:
                return result
        return None
    
    # def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
    #     """Main processing pipeline"""
    #     total_start = time.time()
    #     boxes, yolo_time = self.detect_cubes(frame)
        
    #     if not boxes:
    #         return frame, {'status': 'NO_DETECTION', 'total_time': (time.time() - total_start)*1000}
        
    #     results = []
    #     cache_hits = 0
        
    #     for box_info in boxes:
    #         x, y, w, h = box_info['bbox']
    #         center = box_info['center']
            
    #         cached = self._check_cache(center)
    #         if cached:
    #             results.append(cached)
    #             cache_hits += 1
    #             continue
            
    #         roi = frame[y:y+h, x:x+w]
    #         symbol = self.extract_symbol(roi)
    #         label, conf = self.classify_symbol(symbol)
            
    #         result = {
    #             'bbox': [x, y, w, h],
    #             'center': center,
    #             'label': label,
    #             'confidence': conf,
    #             'yolo_conf': box_info['confidence']
    #         }
    #         results.append(result)
    #         self.result_cache[tuple(center)] = result
            
    #         if len(self.result_cache) > 10:
    #             self.result_cache.pop(next(iter(self.result_cache)))
        
    #     total_time = (time.time() - total_start) * 1000
    #     vis_frame = self.draw_results(frame.copy(), results, total_time, yolo_time, cache_hits)
        
    #     return vis_frame, {
    #         'status': 'OK',
    #         'total_time': total_time,
    #         'yolo_time': yolo_time,
    #         'cache_hits': cache_hits,
    #         'num_detections': len(results)
    #     }
    #==================================fix process_frame==================================
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Xử lý khung hình: Detect -> Crop -> Classify -> Draw
        """
        total_start = time.time()
        
        # 1. Detect (YOLO)
        boxes, yolo_time = self.detect_cubes(frame)
        
        # 2. Xử lý từng box
        results = []
        extract_times = []
        classify_times = []
        
        for box_data in boxes:
            bbox = box_data['bbox']
            x, y, w, h = bbox
            
            # Cắt ảnh (ROI)
            roi = frame[y:y+h, x:x+w]
            
            # a. Kiểm tra Cache (Nếu vị trí không đổi thì lấy kết quả cũ)
            cached_result = self._check_cache(box_data['center'])
            if cached_result:
                label = cached_result['label']
                conf = cached_result['conf']
                color = (0, 255, 0) if label == 'REAL' else (0, 0, 255)
            else:
                # b. Nếu không có cache -> Chạy TinyCNN
                # Extract Symbol
                symbol = self.extract_symbol(roi)
                
                # Classify
                label, conf = self.classify_symbol(symbol)
                
                # Lưu vào cache
                self.result_cache[tuple(box_data['center'])] = {'label': label, 'conf': conf}
                color = (0, 255, 0) if label == 'REAL' else (0, 0, 255)
            
            # Vẽ lên hình
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        total_time = (time.time() - total_start) * 1000
        
        # 3. Đóng gói Metrics (ĐÂY LÀ PHẦN QUAN TRỌNG ĐỂ KHÔNG BỊ LỖI)
        metrics = {
            'status': 'OK',
            'total_time': total_time,
            'num_detections': len(boxes), # <--- Dòng này quan trọng nhất
            'yolo_time': yolo_time
        }
        
        return frame, metrics
    def draw_results(self, frame: np.ndarray, results: List[Dict], total_time: float, 
                    yolo_time: float, cache_hits: int) -> np.ndarray:
        """Draw results + metrics"""
        for res in results:
            x, y, w, h = res['bbox']
            label, conf = res['label'], res['confidence']
            color = (0, 255, 0) if label == 'REAL' else (0, 0, 255)
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{label}: {conf:.2f}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Metrics overlay
        y_offset = 25
        fps = self.calculate_fps(total_time)
        metrics = [
            f"Total: {total_time:.0f}ms",
            f"YOLO: {yolo_time:.0f}ms", 
            f"FPS: {fps:.1f}",
            f"Cache: {cache_hits}/{len(results)}"
        ]
        
        for i, metric in enumerate(metrics):
            cv2.putText(frame, metric, (10, y_offset + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def calculate_fps(self, frame_time: float) -> float:
        self.fps_queue.append(frame_time)
        avg_time = np.mean(self.fps_queue)
        return 1000.0 / avg_time if avg_time > 0 else 0
    
    def print_statistics(self):
        """Print performance stats"""
        logger.info("\n" + "="*50)
        logger.info("PERFORMANCE STATISTICS")
        logger.info("="*50)
        for stage, times in self.timing_stats.items():
            if times:
                avg, std = np.mean(times), np.std(times)
                logger.info(f"{stage.upper()}: {avg:.1f}ms ±{std:.1f}")
        logger.info("="*50 + "\n")


# Utility functions
def setup_camera(camera_id=0, width=640, height=480):
    """Optimized camera setup"""
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap