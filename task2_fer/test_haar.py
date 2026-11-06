"""Test OpenCV Haar cascade face detection on FER images."""
import os, cv2, numpy as np
from PIL import Image

BASE = r"C:\Users\古恩宇\Downloads\2026task2\fer_data\fer_data"
TRAIN_DIR = os.path.join(BASE, "train")
LABEL_MAP = {"Angry":0,"Fear":1,"Happy":2,"Sad":3,"Surprise":4,"Neutral":5}

# Load Haar cascade
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

detected = 0
total = 0
samples = []
for folder in LABEL_MAP:
    d = os.path.join(TRAIN_DIR, folder)
    files = sorted(os.listdir(d))[:50]  # test 50 per class
    for f in files:
        with Image.open(os.path.join(d,f)) as im:
            arr = np.asarray(im.convert("L"), dtype=np.uint8)
        # upscale for detection
        big = cv2.resize(arr, (192,192), interpolation=cv2.INTER_LINEAR)
        faces = face_cascade.detectMultiScale(big, scaleFactor=1.1, minNeighbors=3, minSize=(40,40))
        total += 1
        if len(faces) > 0:
            detected += 1
            if len(samples) < 3:
                samples.append((folder, f, faces[0]))

print(f"detection rate: {detected}/{total} = {detected/total:.1%}")
for s in samples:
    print(f"  sample: {s[0]}/{s[1]} bbox(192px)={s[2]}")
