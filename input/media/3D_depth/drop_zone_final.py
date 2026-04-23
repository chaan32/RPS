import os
import cv2
import numpy as np
import time
import math
from ultralytics import YOLO
from openni import openni2
from dotenv import load_dotenv
from filterpy.kalman import KalmanFilter

class KalmanBoxTracker:
    def __init__(self, initial_x, initial_y, initial_z):
        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        self.kf.x = np.array([initial_x, initial_y, initial_z, 0., 0., 0.])
        self.kf.F = np.eye(6)
        self.kf.H = np.array([[1., 0., 0., 0., 0., 0.],
                              [0., 1., 0., 0., 0., 0.],
                              [0., 0., 1., 0., 0., 0.]])
        self.kf.R *= 10.0
        self.kf.P *= 1000.0 
        
        self.kf.Q = np.eye(6) * 2.0 

    def update_and_predict(self, cx, cy, cz, dt):
        self.kf.F[0, 3] = dt
        self.kf.F[1, 4] = dt
        self.kf.F[2, 5] = dt
        self.kf.predict()
        self.kf.update([cx, cy, cz])
        return self.kf.x.flatten()

def draw_floor_ellipses(image, center, axes_list, colors, alphas):
    for axes, color, alpha in zip(reversed(axes_list), reversed(colors), reversed(alphas)):
        overlay = image.copy()
        cv2.ellipse(overlay, center, axes, 0, 0, 360, color, -1)
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return image


def main():
    load_dotenv()
    openni_dir = os.getenv("OPENNI2_DIR")
    openni2.initialize(openni_dir)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "best_final.pt")
    
    print(f"🚀 SafeAI Kalman Drop-Zone model loading... ({model_path})")
    model = YOLO(model_path) 


    dev = openni2.Device.open_any()
    depth_stream = dev.create_depth_stream()
    color_stream = dev.create_color_stream()

    depth_stream.start()
    color_stream.start()
    dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)

    MIN_DT = 0.01  
    MAX_MERGE_DISTANCE = 150  
    G_MM = 9800.0  

    kf_trackers = {}
    prev_time = time.time()

    # 타원 설정을 위한 데이터
    radii_mm = [200, 400, 800] # 위험 20cm, 주의 40cm, 안전 80cm
    colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0)] # 빨강, 노랑, 초록
    alphas = [0.6, 0.4, 0.2] # 투명도 설정

    CAMERA_HEIGHT_MM = 1700.0  
    CAMERA_TILT_DEG = 22.0  
    tilt_rad = math.radians(CAMERA_TILT_DEG)

    while True:
        depth_frame = depth_stream.read_frame()
        color_frame = color_stream.read_frame()
        
        depth_data = depth_frame.get_buffer_as_uint16()
        depth_array = np.ndarray((depth_frame.height, depth_frame.width), dtype=np.uint16, buffer=depth_data)
        
        color_data = color_frame.get_buffer_as_uint8()
        color_array = np.ndarray((color_frame.height, color_frame.width, 3), dtype=np.uint8, buffer=color_data)
        color_image = cv2.cvtColor(color_array, cv2.COLOR_RGB2BGR)

        current_time = time.time()
        dt = current_time - prev_time
        prev_time = current_time

       
        results = model.track(color_image, persist=True, tracker="botsort.yaml", conf=0.35, verbose=False)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.int().cpu().numpy()
            
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
                
                class_name = model.names[cls_id] 
                if class_name not in ["box_1", "box_2"]:
                    continue

                x1, y1, x2, y2 = box
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                z_depth = depth_array[cy, cx]

                if z_depth == 0 or z_depth > 10000:
                    continue 

                if track_id not in kf_trackers:
                    closest_old_id = None
                    min_dist = float('inf')
                    for old_id, tracker_obj in list(kf_trackers.items()):
                        old_x, old_y = tracker_obj.kf.x[0], tracker_obj.kf.x[1]
                        dist = np.linalg.norm(np.array([cx, cy]) - np.array([old_x, old_y]))
                        if dist < min_dist and dist < MAX_MERGE_DISTANCE:
                            min_dist = dist
                            closest_old_id = old_id
                    
                    if closest_old_id is not None:
                        kf_trackers[track_id] = kf_trackers.pop(closest_old_id)
                    else:
                        kf_trackers[track_id] = KalmanBoxTracker(cx, cy, z_depth)

                if dt > MIN_DT:
                    filtered_state = kf_trackers[track_id].update_and_predict(cx, cy, z_depth, dt)
                    
                    smooth_cx, smooth_cy, smooth_z = filtered_state[0], filtered_state[1], filtered_state[2]
                    vx, vy, vz = filtered_state[3], filtered_state[4], filtered_state[5]

                    if abs(vx) < 15: vx = 0
                    if abs(vy) < 15: vy = 0

                    OPTICAL_SCALE = 800.0  
                    vertical_dist_to_box = smooth_z * math.sin(tilt_rad)
                    h_fall = CAMERA_HEIGHT_MM - vertical_dist_to_box

                    if h_fall > 0:
                        t_fall = math.sqrt((2 * h_fall) / G_MM)
                        pixel_drop_y = int((h_fall * math.cos(tilt_rad) * OPTICAL_SCALE) / smooth_z)

                        drop_x = int(smooth_cx + (vx * t_fall))
                        drop_y = y2 + pixel_drop_y  

                        drop_x = max(0, min(color_image.shape[1], drop_x))
                        drop_y = max(0, min(color_image.shape[0] - 10, drop_y))

                        axes_list = []
                        for mm_radius in radii_mm:
                            axis_x = int(mm_radius * OPTICAL_SCALE / smooth_z) 
                            axis_y = int(axis_x * math.sin(tilt_rad)) 
                            axis_y = max(axis_y, 5) 
                            axes_list.append((axis_x, axis_y))

                        color_image = draw_floor_ellipses(color_image, (drop_x, drop_y), axes_list, colors, alphas)

                        cv2.line(color_image, (int(smooth_cx), int(y2)), (drop_x, drop_y), (0, 165, 255), 2)
                        
                        inner_axis_x = axes_list[0][0]
                        inner_axis_y = axes_list[0][1]
                        info_text = f"{class_name.upper()} KF ({t_fall:.1f}s)" 
                        cv2.putText(color_image, info_text, (drop_x - 60, drop_y - inner_axis_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                        cv2.line(color_image, (drop_x - 10, drop_y), (drop_x + 10, drop_y), (0, 0, 255), 2)
                        cv2.line(color_image, (drop_x, drop_y - 5), (drop_x, drop_y + 5), (0, 0, 255), 2)

                # 현재 상자 시각화 (녹색 네모)
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(color_image, (cx, cy), 5, (0, 255, 0), -1)
                cv2.putText(color_image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("SafeAI - Kalman Drop-Zone Engine", color_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    depth_stream.stop()
    color_stream.stop()
    openni2.unload()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()