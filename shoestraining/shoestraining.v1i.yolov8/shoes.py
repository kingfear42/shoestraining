import os
from ultralytics import YOLO

base_dir = r"C:\Users\krkmz\Desktop\Aware Robotics Opencv\shoestraining\shoestraining.v1i.yolov8"

model_path = os.path.join(base_dir, r"C:\Users\krkmz\Desktop\Aware Robotics Opencv\shoestraining\shoestraining.v1i.yolov8\runs\segment\yolov8_custom_model\weights\best.pt") 

source_path = os.path.join(base_dir, r"C:\Users\krkmz\Desktop\Aware Robotics Opencv\shoestraining\shoestraining.v1i.yolov8\test\images\taban2.jpg") 

print("Fotoğraf işleniyor ve ekranda gösteriliyor...")

model = YOLO(model_path)
results = model.predict(source=source_path, conf=0.05, show=True, save=True)