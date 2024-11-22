
import torch
import os
from .infer_image import infer, get_align
from torch.nn.modules.distance import PairwiseDistance
import cv2
from PIL import Image
from .get_embedding import load_embeddings_and_names
from .getface import yolo
import torch.nn.functional as F
from collections import Counter
import numpy as np
from models.spoofing.FasNet import Fasnet
from .utils import get_model
from collections import defaultdict


l2_distance = PairwiseDistance(p=2)
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
antispoof_model = Fasnet()


def find_closest_person(pred_embed, embeddings, image2class, distance_mode='cosine', l2_threshold=1, cosine_threshold=0.5):
    """
    Hàm tính toán khoảng cách trung bình giữa pred_embed và các lớp trong cơ sở dữ liệu và trả về lớp gần nhất.
    """
    # Chuyển embeddings thành tensor
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)

    # Tính toán khoảng cách giữa pred_embed và tất cả embeddings
    if distance_mode == 'l2':
        # Tính khoảng cách L2 (Euclidean distance)
        distances = torch.norm(embeddings_tensor - pred_embed, dim=1).detach().cpu().numpy()
    else:
        # Tính độ tương đồng Cosine chuyển thành khoảng cách
        similarities = F.cosine_similarity(pred_embed, embeddings_tensor)
        distances = (1 - similarities).detach().cpu().numpy()

    # Chuyển image2class thành mảng NumPy để xử lý nhanh
    image2class_np = np.array([image2class[i] for i in range(len(embeddings))])
    
    # Sử dụng NumPy để tính tổng khoảng cách và số lượng phần tử mỗi lớp
    num_classes = max(image2class.values()) + 1
    
    # Tính tổng khoảng cách cho mỗi lớp
    total_distances = np.zeros(num_classes, dtype=np.float32)
    np.add.at(total_distances, image2class_np, distances)
    
    # Tính số lượng phần tử mỗi lớp
    counts = np.zeros(num_classes, dtype=np.int32)
    np.add.at(counts, image2class_np, 1)

    # Tính trung bình khoảng cách cho mỗi lớp (tránh chia cho 0)
    avg_distances = np.divide(total_distances, counts, out=np.full_like(total_distances, np.inf), where=counts > 0)

    if distance_mode == 'l2':
        best_class = np.argmin(avg_distances)  # Lớp có khoảng cách trung bình nhỏ nhất
        if avg_distances[best_class] < l2_threshold:
            return best_class
    else:  # Cosine
        best_class = np.argmin(avg_distances)  # Với Cosine, khoảng cách nhỏ nhất là tốt nhất
        if avg_distances[best_class] < cosine_threshold:
            return best_class
    
    return -1
 


def find_closest_person_vote(pred_embed, embeddings, image2class, distance_mode= 'cosine', k=15, vote_threhold = 0.8, l2_threshold= 1, cosine_threshold= 0.5 ):

    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    if distance_mode == 'l2':
        distances = torch.norm(embeddings_tensor - pred_embed, dim=1).detach().cpu().numpy()
        for distance in np.sort(distances)[:k]:
            if distance > l2_threshold:
                return -1
    else:
        similarities = F.cosine_similarity(pred_embed, embeddings_tensor)
        distances = (1 - similarities).detach().cpu().numpy() 
        for distance in  np.sort(distances)[:k]:
            if distance > cosine_threshold:
                return -1
    
    k_smallest_indices = np.argsort(distances)[:k]
    k_nearest_classes = [image2class[idx] for idx in k_smallest_indices]
    class_counts = Counter(k_nearest_classes)

    best_class_index = class_counts.most_common(1)[0][0]

    if k_nearest_classes.count(best_class_index) >= vote_threhold * len(k_nearest_classes):
        return best_class_index
    else:
        return -1
    

if __name__ == '__main__':

    recogn_model_name= 'inceptionresnetV1'
    
    embedding_file_path= 'data/data_source/inceptionresnetV1_embeddings.npy'
    image2class_file_path = 'data/data_source/inceptionresnetV1_image2class.pkl'
    index2class_file_path = 'data/data_source/inceptionresnetV1_index2class.pkl'

    embeddings, image2class, index2class = load_embeddings_and_names(embedding_file_path, image2class_file_path, index2class_file_path)


    recogn_model = get_model(recogn_model_name)
  
    test_folder = 'testdata/unknown'
    for i in os.listdir(test_folder):
        image_path = os.path.join(test_folder, i)
        image = Image.open(image_path).convert('RGB')
        align_image, faces, probs, lanmark  = get_align(image)
        pred_embed= infer(recogn_model, align_image)
        result = find_closest_person(pred_embed, embeddings, image2class)
        print(result)
 