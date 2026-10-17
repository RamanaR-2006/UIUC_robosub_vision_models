from ultralytics import YOLO

# choose pretrained model yolo 26 nano
model = YOLO("yolo26n.pt")

# train a model on the dataset. epochs = 100 means it makes 100 different passes
# imgsz 640 means it puts images to pixel size 640 (cuz i dont want laptop to explode)
# device = mps -> use gpu
results = model.train(data="card_detection/dataset/data.yaml", epochs=100, imgsz=640, device="mps")