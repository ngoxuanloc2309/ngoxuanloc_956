import cv2
import numpy as np
import os

output_folder = 'Ket_qua'
os.makedirs(output_folder, exist_ok=True)
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
classes = []
with open('yolov3.txt', 'r') as f:
    classes = f.read().splitlines()
dem = {class_name: 0 for class_name in classes}

while True:
    input_image_path = input("Nhập tên ảnh đầu vào hoặc nhập 'exit' để thoát: ")

    if input_image_path.lower() == 'exit':
        break

    img = cv2.imread(input_image_path)
    if img is None:
        print(f"Không thể đọc ảnh từ '{input_image_path}'")
        continue

    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)

    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    print(f"Đã phát hiện {len(indexes)} đối tượng:")
    for i in indexes.flatten():
        print(f" - {classes[class_ids[i]]} với độ chắc chắn {confidences[i]}")

    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]

        # Tạo thư mục 'label' nếu chưa tồn tại
        label_folder = os.path.join(output_folder, label)
        os.makedirs(label_folder, exist_ok=True)

        # Cắt đối tượng từ ảnh gốc
        object_roi = img[y:y+h, x:x+w]

        # Lưu ảnh cắt được vào thư mục tương ứng
        filename = f"{label}_{dem[label] + 1}.jpg"
        file_path = os.path.join(label_folder, filename)
        cv2.imwrite(file_path, object_roi)

        dem[label] += 1
    print(f"Đã lưu đối tượng vào thư mục '{label_folder}'")
