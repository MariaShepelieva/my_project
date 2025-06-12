import cv2 
import torch
import numpy as np
from src.utils.stream_utils import GazeLabelAggregator
from src.model import EyeClassifierCNN
from src.dataset import EyeDataset
from src.augmentations import train_transforms

# Загрузка модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EyeClassifierCNN(num_classes=4)
model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
model.to(device)
model.eval()

# Классы и агрегатор
classes = EyeDataset("data", (224, 224)).classes
aggregator = GazeLabelAggregator(window_size=30)

# Счётчики
counts = [0] * len(classes)
total = 0

# Детекторы лица и глаз
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
frame_interval = int(fps)
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if frame_idx % frame_interval == 0:
        faces = face_cascade.detectMultiScale(gray_full, scaleFactor=1.1, minNeighbors=5)
        probs_accum = []

        for (x, y, w, h) in faces:
            roi_gray = gray_full[y:y+h, x:x+w]
            eyes = eyes_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
            for (ex, ey, ew, eh) in eyes:
                eye_crop = roi_gray[ey:ey+eh, ex:ex+ew]
                resized = cv2.resize(eye_crop, (224, 224))[:, :, None]
                inp = train_transforms(image=resized)['image'].unsqueeze(0).to(device)
                with torch.no_grad():
                    out = model(inp)
                    prob = torch.softmax(out, dim=1).cpu().numpy()[0]
                    probs_accum.append(prob)

        if probs_accum:
            avg_prob = np.mean(probs_accum, axis=0)
            max_prob = np.max(avg_prob)
            max_idx = int(np.argmax(avg_prob))

            if max_prob >= 0.6:
                pred = max_idx
            else:
                close_prob = avg_prob[0]
                left_prob = avg_prob[2]
                right_prob = avg_prob[3]

                if close_prob < 0.5 and left_prob < 0.5 and right_prob < 0.5:
                    pred = 1  # forward
                else:
                    pred = 0  # close

                avg_prob = [0.0] * len(classes)
                avg_prob[pred] = 1.0
        else:
            pred = 0  # Очі не знайдено — вважаємо закритими
            avg_prob = [0.0] * len(classes)
            avg_prob[pred] = 1.0

        counts[pred] += 1
        total += 1
        aggregator.update(pred)

    if 'faces' in locals():
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    if 'eyes' in locals():
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)

    label, conf = aggregator.get_dominant_label()
    text = f"{classes[label]} ({conf*100:.1f}%)" if label is not None else "Calculating..."
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Eye Gaze Aggregation", frame)
    frame_idx += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
