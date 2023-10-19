from ultralytics import YOLO

import cv2

import time

def fun():
    model = YOLO('yolov8n.pt')
    
    crop_percentage = 1# 0.45

    # video_path = "videos/bright_3.mp4"
    cap = cv2.VideoCapture(0)

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * crop_percentage)

    fps_time = time.perf_counter()
    counter = 0
    fps = 0

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model.predict(frame, device='cpu')

            annotated_frame = results[0].plot()

            counter += 1

            if(time.perf_counter() - fps_time > 1):
                fps = int(counter / (time.perf_counter() - fps_time))
                fps_time = time.perf_counter()
                counter = 0

            cv2.putText(annotated_frame, str(fps), (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow("YOLOv8 Inference", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

fun()