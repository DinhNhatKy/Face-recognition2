
import torch
import os
from .infer_image import infer
from torch.nn.modules.distance import PairwiseDistance
import cv2
from PIL import Image
from .get_embedding import load_embeddings_and_names
from .getface import yolo
import torch.nn.functional as F
from collections import Counter
import numpy as np



l2_distance = PairwiseDistance(p=2)
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')


def find_closest_person(pred_path, embeddings, names, recogn_model_name, distance_mode):
    image = Image.open(pred_path).convert('RGB')
    pred_embed = infer(recogn_model_name, image)
    if isinstance(pred_embed, torch.Tensor):
        pred_embed = pred_embed.cpu()

    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)


    total_distances = {}
    counts = {}
    distances = None

    if distance_mode == 'cosine':
        distances = F.cosine_similarity(pred_embed, embeddings_tensor)
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

    name_of_person = 'Unknown'

    if distance_mode =='l2':
        name_of_person = min(avg_distances, key=avg_distances.get)
    else: 
        name_of_person = max(avg_distances, key=avg_distances.get)

    
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


def find_closest_person_vote(pred_path, embeddings, names, recogn_model_name, distance_mode, k=5):
    image = Image.open(pred_path).convert('RGB')
    pred_embed = infer(recogn_model_name, image)
    if isinstance(pred_embed, torch.Tensor):
        pred_embed = pred_embed.cpu()

    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)

    if distance_mode == 'cosine':
        distances = F.cosine_similarity(pred_embed, embeddings_tensor).cpu().detach().numpy()
    else:
        distances = torch.norm(embeddings_tensor - pred_embed, dim=1).detach().cpu().numpy().cpu().detach().numpy()

    # Find the k smallest distances
    if distance_mode == 'l2':
        k_smallest_indices = np.argsort(distances)[:k]
    else:
        k_smallest_indices = np.argsort(-distances)[:k]  # for cosine similarity, larger is closer

    # Get the names associated with the k smallest distances
    k_nearest_names = [names[i] for i in k_smallest_indices]

    # Vote for the most common class
    most_common_name = Counter(k_nearest_names).most_common(1)[0][0]

    img = cv2.imread(pred_path)
    results = yolo(img)
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        x1, y1, x2, y2 = map(int, boxes[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
        cv2.putText(img, most_common_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Face Recognition', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return most_common_name


if __name__ == '__main__':

    recogn_model_name= 'resnet34'
    test_folder_path = 'testdata/thaotam'
    embedding_file_path = f'data/embedding_names/{recogn_model_name}_embeddings.npy'
    names_file_path = f'data/embedding_names/{recogn_model_name}_names.pkl'
   
    embeddings, names = load_embeddings_and_names(embedding_file_path, names_file_path)


    for file_name in os.listdir(test_folder_path):
        pred_path = os.path.join(test_folder_path, file_name)
        name_of_person = find_closest_person_vote(pred_path, embeddings, names, recogn_model_name, 'cosine', 10)
   

   
