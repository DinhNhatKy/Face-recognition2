import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from models.face_detect.mtcnn import MTCNN
import torch

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mtcnn_inceptionresnetV1 = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    select_largest= True,
    selection_method= 'largest',
    device=device,
    
)

mtcnn_resnet = MTCNN(
    image_size=224, margin=0, min_face_size=20,
    device=device
)

model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
yolo = YOLO(model_path)

if __name__ == '__main__':
    
    from PIL import Image
    image_path = 'testdata/sontung/002.jpg'
    image = Image.open(image_path).convert('RGB')
    boxes, probs = mtcnn_inceptionresnetV1.detect(image)
    print(boxes)



