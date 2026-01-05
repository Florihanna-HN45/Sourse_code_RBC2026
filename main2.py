import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import logging
from util2 import OptimizedCubeDetector, setup_camera

# Config logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main application loop"""
    
    # CONFIGURATION - DỄ THAY ĐỔI
    CONFIG = {
        'yolo_model': r'D:/1. STUDY/ROBOCON 2026/04012026/models/best.pt',        # Your trained model
        'tflite_model': r'D:/1. STUDY/ROBOCON 2026/04012026/models/final.tflite',  # Your CNN model
        'confidence_threshold': 0.7,
        'camera_id': 0
    }
    
    logger.info("🎯 Starting Cube Authenticity Detector")
    logger.info(f"📁 YOLO: {CONFIG['yolo_model']}")
    logger.info(f"🧠 TFLite: {CONFIG['tflite_model']}")
    
    try:
        # Initialize detector
        detector = OptimizedCubeDetector(
            yolo_model_path=CONFIG['yolo_model'],
            tflite_model_path=CONFIG['tflite_model'],
            confidence_threshold=CONFIG['confidence_threshold']
        )
        
        # Setup camera
        cap = setup_camera(CONFIG['camera_id'])
        if not cap.isOpened():
            logger.error("❌ Cannot open camera!")
            return
        
        logger.info("🎥 Camera ready | 'q'=quit, 's'=stats")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to grab frame")
                break
            
            # Process frame
            result_frame, metrics = detector.process_frame(frame)
            
            # Display
            cv2.imshow('Cube Detector [RPi3 Ready]', result_frame)
            
            # # Periodic logging
            # frame_count += 1
            # if frame_count % 30 == 0:
            #     status = metrics['status']
            #     time_ms = metrics['total_time']
            #     detections = metrics['num_detections']
            #     logger.info(f"Frame {frame_count} | {status} | {time_ms:.1f}ms | {detections} cubes")
            #============ ĐỂ CHƯƠNG TRÌNH KHÔNG STOP KHI THIẾU KEY 'num_detections' ============#
            # Periodic logging
            frame_count += 1
            if frame_count % 30 == 0:
                # Dùng .get() để nếu không có key thì trả về giá trị mặc định, không báo lỗi
                status = metrics.get('status', 'Running') 
                time_ms = metrics.get('total_time', 0.0)
                detections = metrics.get('num_detections', 0) # <--- SỬA DÒNG NÀY
                
                logger.info(f"Frame {frame_count} | {status} | {time_ms:.1f}ms | {detections} cubes")
            
            # Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("User quit")
                break
            elif key == ord('s'):
                detector.print_statistics()
    
    except KeyboardInterrupt:
        logger.info("⏹️  Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        logger.info("✅ Application shutdown complete")


if __name__ == "__main__":
    main()