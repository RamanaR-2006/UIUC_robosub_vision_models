from ultralytics import YOLO

# windows starts dataloader workers by re-importing this file, so training
# has to live under the main guard or each worker tries to start training again
if __name__ == "__main__":
    # choose pretrained model yolo 26 nano
    model = YOLO("yolo26n.pt")

    # train a model on the dataset. epochs = 100 means it makes 100 different passes
    # imgsz 640 means it puts images to pixel size 640 (cuz i dont want laptop to explode)
    # device = cuda -> use the nvidia gpu
    results = model.train(data="card_detection/dataset/data.yaml", epochs=50, imgsz=640, batch=16, patience = 10, time = 1, device="cuda", workers=4)
