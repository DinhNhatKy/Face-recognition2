from torchvision import datasets
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import os
from .infer_image import infer
from torch.nn.modules.distance import PairwiseDistance
import pickle
import cv2
from PIL import Image
from supervision import Detections
from .get_embedding import load_embeddings_and_names
from .getface import yolo
from sklearn.metrics.pairwise import cosine_similarity


l2_distance = PairwiseDistance(p=2)
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')


def find_closest_person(pred_path, embeddings, names, recogn_model_name, distance_mode):
    pred_embed = infer(recogn_model_name, pred_path)
    if isinstance(pred_embed, torch.Tensor):
        pred_embed = pred_embed.cpu()

    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)

    total_distances = {}
    counts = {}

    if distance_mode == 'cosine':
        similarity = cosine_similarity()
        distances = torch.norm(embeddings_tensor - pred_embed, dim=1).detach().cpu().numpy()


    else:
        distances = torch.norm(embeddings_tensor - pred_embed, dim=1).detach().cpu().numpy()

    for i, name in enumerate(names):
        
        distance = distances[i]
        if name not in total_distances:
            total_distances[name] = 0
            counts[name] = 0
        total_distances[name] += distance
        counts[name] += 1

    avg_distances = {name: total_distances[name] / counts[name] for name in total_distances}

    name_of_person = min(avg_distances, key=avg_distances.get)
    
    img = cv2.imread(pred_path)
    results = yolo(img)
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        x1, y1, x2, y2 = map(int, boxes[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
        cv2.putText(img, name_of_person, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Face Recognition', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return avg_distances, name_of_person


if __name__ == '__main__':

    recogn_model_name= 'inceptionresnetV1'
    test_folder_path = 'testdata/chipu'
    embedding_file_path = f'data/embedding_names/{recogn_model_name}_embeddings.npy'
    names_file_path = f'data/embedding_names/{recogn_model_name}_names.pkl'
   
    embeddings, names = load_embeddings_and_names(embedding_file_path, names_file_path)
    print(embeddings.shape)


    for file_name in os.listdir(test_folder_path):
        pred_path = os.path.join(test_folder_path, file_name)
        find_closest_person(pred_path, embeddings, names, recogn_model_name)
      
