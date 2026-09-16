from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")

results = model.train(
    data=r"C:\Users\krkmz\Desktop\Aware Robotics Opencv\shoestraining\shoestraining.v1i.yolov8\data.yaml",
    epochs=50,      
    imgsz=640,    
    batch=16,        
    workers=2,
    device='cpu',       
    name="yolov8_custom_model" 
)
