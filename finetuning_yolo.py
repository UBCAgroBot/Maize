from ultralytics import YOLO
# Load a pretrained model
model = YOLO(model='yolov8n.pt')

# Train the model
model.train(data="config.yaml", epochs=1, imgsz=640)
metrics = model.val()
path = model.export(format="onnx")  # export the model to ONNX format1~path = model.export(format="onnx")  # export the model to ONNX format