import cv2
import numpy as np

# Mapeamento das classes COCO
classes = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "item", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet", "TV monitor", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

def load_yolov7():
    # Carregue o modelo YOLOv7
    net = cv2.dnn.readNet("yolov7.weights", "yolov7.cfg")
    
    # Obtenha os nomes das camadas
    layer_names = net.getLayerNames()
    
    # Obtenha as saídas não conectadas
    out_layers = net.getUnconnectedOutLayers()
    
    if isinstance(out_layers[0], np.ndarray):
        out_layers = [i[0] - 1 for i in out_layers]
    else:
        out_layers = [i - 1 for i in out_layers]
    
    output_layers = [layer_names[i] for i in out_layers]
    
    return net, output_layers

def detect_objects(frame, net, output_layers):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            detection = detection.flatten()  # Flattens the array for easier access
            scores = detection[5:]
            if scores.size == 0:
                continue

            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    detected_objects = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            detected_objects.append((class_ids[i], confidences[i], (x, y, w, h)))

            # Desenhe o quadrado verde para pessoas e amarelo para outros itens
            if class_ids[i] == 0:  # Person
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {confidences[i]:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:  # Other items
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Yellow color (BGR format)
                cv2.putText(frame, f"{classes[class_ids[i]]} {confidences[i]:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return frame, detected_objects

def detect_person_picking_item(frame, detected_objects):
    person_boxes = [box for class_id, _, box in detected_objects if class_id == 0]  # Pessoas
    item_boxes = [box for class_id, _, box in detected_objects if class_id != 0]  # Outros itens

    for (px, py, pw, ph) in person_boxes:
        hand_region = (px, py + int(ph / 2), pw, int(ph / 2))
        for (ix, iy, iw, ih) in item_boxes:
            if (hand_region[0] < ix < hand_region[0] + hand_region[2] and 
                hand_region[1] < iy < hand_region[1] + hand_region[3]):
                cv2.rectangle(frame, (ix, iy), (ix + iw, iy + ih), (0, 0, 255), 2)
                cv2.putText(frame, "Item Picked!", (ix, iy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return frame

def main():
    net, output_layers = load_yolov7()  # Corrigido para chamar a função correta
    cap = cv2.VideoCapture("video.mp4")

    if not cap.isOpened():
        print("Erro ao abrir o vídeo. Verifique o caminho e o formato do arquivo.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, detected_objects = detect_objects(frame, net, output_layers)
        frame = detect_person_picking_item(frame, detected_objects)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
