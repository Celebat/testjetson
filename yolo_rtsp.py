import os
import cv2
import numpy as np
from ultralytics import YOLO
import time

# Загрузка модели YOLOv8m TensorRT или экспорт модели, если она отсутствует
engine_path = "yolov8l.engine"
if not os.path.exists(engine_path):
    print(f"Модель {engine_path} не найдена. Экспорт модели в TensorRT формат...")
    model = YOLO("yolov8l.pt")
    model.export(format="engine")

# Загрузка модели YOLOv8m TensorRT
trt_model = YOLO(engine_path)

# Установка параметров конфиденс и imgsz
conf_threshold = 0.5
imgsz = 640

rtsp_url = 'rtsp://jetson:jv4wf4f5@192.168.0.103:554/Streaming/Channels/101'

rois = [
    [(875, 92), (1037, 87), (1073, 203), (789, 222)],  # Зона 1
    [(789, 222), (1073, 203), (1111, 350), (670, 400)],  # Зона 2
    [(640, 442), (888, 555), (875, 804), (505, 825)],  # Зона 3
    [(888, 555), (1135, 534), (1192, 788), (875, 804)]   # Зона 4
]

# Инициализация счетчиков и времени
people_in_zone = [0] * len(rois)
total_people_passed = [0] * len(rois)
last_detection_time = [0] * len(rois)
roi_active_timer = [0] * len(rois)
exit_delay = 1.0  # Время, по истечении которого человек считается вышедшим

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

    # Выполнение инференса на текущем кадре с заданным размером изображения и порогом уверенности
    results = trt_model(frame, conf=conf_threshold, imgsz=imgsz)

    # Инициализация списка для хранения статуса подсветки зон и счетчиков текущего кадра
    roi_active = [False] * len(rois)
    people_in_current_frame = [0] * len(rois)

    # Текущее время
    current_time = time.time()

    # Обработка каждого результата
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)

            if box.conf[0] >= conf_threshold and box.cls[0] == 0:  # 0 - класс 'person' в COCO
                roi_index = is_in_roi((center_x, center_y), rois)
                if roi_index != -1:
                    roi_active[roi_index] = True
                    people_in_current_frame[roi_index] += 1
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    # Обновление счетчиков и времени обнаружения
    for i, roi in enumerate(rois):
        if roi_active[i]:
            # Если человек обнаружен в зоне, обновляем таймер
            last_detection_time[i] = current_time
            roi_active_timer[i] = 0
        else:
            # Если человек не обнаружен, увеличиваем таймер неактивности
            roi_active_timer[i] = current_time - last_detection_time[i]

        if roi_active_timer[i] > exit_delay and people_in_zone[i] > 0:
            # Человек считается вышедшим, если не обнаружен в зоне дольше, чем exit_delay
            people_in_zone[i] = 0
        elif roi_active[i] and people_in_zone[i] == 0:
            # Если человек обнаружен в зоне после выхода, увеличиваем счётчик
            total_people_passed[i] += 1
            people_in_zone[i] = people_in_current_frame[i]

        # Подсветка зон интереса
        color = (0, 0, 255) if roi_active[i] else (255, 0, 0)
        cv2.polylines(frame, [np.array(roi, dtype=np.int32)], True, color, 2)

        # Подпись зоны
        cv2.putText(frame, f"Zone {i+1}", (roi[0][0], roi[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Отображение количества людей в зоне и общего количества
        text = f"People now: {people_in_zone[i]} | Total: {total_people_passed[i]}"
        cv2.putText(frame, text, (roi[0][0], roi[0][1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Показать результат
    cv2.imshow('RTSP Stream', frame)

    # Остановка по клавише 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
