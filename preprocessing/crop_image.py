from facenet_pytorch import MTCNN, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import os
from models.mtcnn import MTCNN

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.5, 0.65, 0.65], factor=0.709, post_process=True
)

batch_size = 8
data_dir = 'data/dataset'
workers = 0 if os.name == 'nt' else 8

dataset = datasets.ImageFolder(data_dir, transform=transforms.Resize((512, 512)))
dataset.samples = [
    (p, p.replace(data_dir, data_dir + '_cropped'))
        for p, _ in dataset.samples
]

loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    collate_fn=training.collate_pil
)

for i, (x, y) in enumerate(loader):
    boxes, probs = mtcnn.detect(x)
    
    if boxes is not None:  
        mtcnn(x, save_path=y)
        print(f'\rBatch {i + 1} of {len(loader)}: Đã lưu ảnh crop', end='')
    else:
        print(f'\rBatch {i + 1} of {len(loader)}: Không phát hiện mặt trong batch', end='')

del mtcnn
