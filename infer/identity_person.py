from torchvision import datasets
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import os
from .getface import get_face_mtcnn, get_face_yolo, yolo,  mtcnn
from .infer_image import get_model, infer
from torch.nn.modules.distance import PairwiseDistance




l2_distance = PairwiseDistance(p=2)
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

workers = 0 if os.name == 'nt' else 4

def create_data_embeddings(data_gallary_path, detection_model, recognition_model):

    def collate_fn(x):
        return x[0]

    dataset = datasets.ImageFolder(data_gallary_path)
    dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

    aligned = []
    names = []

    for x, y in loader:
        x_aligned, prob = detection_model(x, return_prob=True)
        if x_aligned is not None:
            aligned.append(x_aligned)
            names.append(dataset.idx_to_class[y])
    aligned = torch.stack(aligned).to(device)
    embeddings = recognition_model(aligned).detach().to(device)
    return embeddings, names



def find_closest_person(pred_path, embeddings, detect_model_name, recogn_model):
    pred_embed= infer(detect_model_name, recogn_model, pred_path)
        
    scores=[]
    for compare_embed in embeddings:
        score = l2_distance.forward(pred_embed, compare_embed)
        scores.append(score)

    optim_index = np.argmin(scores)

    return optim_index



if __name__ == '__main__':
    
    recognition_model = get_model('inceptionresnetV1')
    data_gallary_path = 'data/dataset'
    embeddings, names = create_data_embeddings(data_gallary_path, mtcnn, recognition_model)
    print(embeddings.shape)
    print(names)

    pred_path = 'testdata/sontung/sontung1.jpg'
    optim_index = find_closest_person(pred_path, embeddings, 'mtcnn', recognition_model)
    print(optim_index)