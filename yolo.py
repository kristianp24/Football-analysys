
from ultralytics import YOLO

model = YOLO('colab_training/best.pt')
rez = model.predict('input/8fd33_4.mp4',conf = 0.5,save=True)
