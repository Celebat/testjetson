import cv2
import numpy as np
from ultralytics import YOLO

# Загрузка модели YOLOv8m TensorRT
trt_model = YOLO("yolov8m.engine")

rtsp_url = 'rtsp://jetson:jv4wf4f5@192.168.0.108:554/Streaming/Channels/101'

rois = [
    [(100, 100), (300, 100), (300, 300), (100, 300)],  # Зона 1
    [(350, 100), (550, 100), (550, 300), (350, 300)],  # Зона 2
    [(100, 350), (300, 350), (300, 550), (100, 550)],  # Зона 3
    [(350, 350), (550, 350), (550, 550), (350, 550)]   # Зона 4
]

# Функция для проверки, находится ли центр объекта в одной из зон интереса
def is_in_roi(center, rois):
    for i, roi in enumerate(rois):
        contour = np.array(roi, dtype=np.int32)
        if cv2.pointPolygonTest(contour, center, False) >= 0:
            return i
    return -1

# Подключение к RTSP камере
cap = cv2.VideoCapture(rtsp_url)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Выполнение инференса на текущем кадре
    results = trt_model(frame)

    # Инициализация списка для хранения статуса подсветки зон
    roi_active = [False] * len(rois)

    # Обработка каждого результата
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # Проверка, находится ли человек в одной из зон интереса
            if box.cls[0] == 0:  # 0 - класс 'person' в COCO
                roi_index = is_in_roi((center_x, center_y), rois)
                if roi_index != -1:
                    roi_active[roi_index] = True
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    # Подсветка зон интереса
    for i, roi in enumerate(rois):
        color = (0, 0, 255) if roi_active[i] else (255, 0, 0)  # Красный, если человек в зоне
        cv2.polylines(frame, [np.array(roi, dtype=np.int32)], True, color, 2)

    # Показать результат
    cv2.imshow('RTSP Stream', frame)

    # Остановка по клавише 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
