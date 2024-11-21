
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


def find_closest_person(pred_embed, embeddings, image2class, distance_mode='cosine'):
    """
    Hàm tính toán khoảng cách trung bình giữa pred_embed và các lớp trong cơ sở dữ liệu và trả về lớp gần nhất.
    
    Parameters:
    - pred_embed (torch.Tensor): Embedding của ảnh cần nhận diện.
    - embeddings (list): Danh sách các embeddings của ảnh trong cơ sở dữ liệu.
    - image2class (dict): Mảng ánh xạ giữa index ảnh và lớp tương ứng (từ 0 đến n_classes-1).
    - distance_mode (str): 'cosine' cho cosine similarity, 'l2' cho L2 distance.

    Returns:
    - avg_distances (list): Mảng chứa khoảng cách trung bình từ pred_embed đến mỗi lớp.
    - best_class (int): Chỉ số lớp gần nhất (có khoảng cách trung bình nhỏ nhất hoặc lớn nhất tùy vào distance_mode).
    """
    # Chuyển embeddings thành tensor
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)

    # Lưu tổng distances và số lượng của mỗi lớp
    total_distances = defaultdict(float)  # Lưu tổng distances cho mỗi class
    counts = defaultdict(int)  # Lưu số lần gặp mỗi class

    # Tính toán distances giữa pred_embed và tất cả các embeddings
    if distance_mode == 'l2':
        distances = torch.norm(embeddings_tensor - pred_embed, dim=1).detach().cpu().numpy()
       
    else:
        similarities = F.cosine_similarity(pred_embed, embeddings_tensor)
        distances = (1 - similarities).detach().cpu().numpy() 

    for i, name in enumerate(embeddings):
        # Lấy class của ảnh dựa trên image2class
        class_label = image2class.get(i, None)

        if class_label is not None:
            total_distances[class_label] += distances[i]
            counts[class_label] += 1

    # Tính toán trung bình distance cho mỗi class
    num_classes = max(image2class.values()) + 1  # Đảm bảo số lớp chính xác
    avg_distances = [(total_distances[class_label] / counts[class_label]).item() if counts[class_label] > 0 else float('inf') 
                     for class_label in range(num_classes)]


    return avg_distances
    if distance_mode == 'l2':
        best_class = min(range(num_classes), key=lambda x: avg_distances[x])
        if avg_distances[best_class] < 1.1 :
            return best_class
    else:
        best_class = max(range(num_classes), key=lambda x: avg_distances[x])
        if avg_distances[best_class] > 0.5:
            return best_class
    return -1
 


def find_closest_person_vote(pred_embed, embeddings, image2class, distance_mode='cosine', k=10, threshold=0.85):
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)

    # Tính khoảng cách
    if distance_mode == 'cosine':
        distances = F.cosine_similarity(pred_embed, embeddings_tensor).cpu().detach().numpy()
        valid_condition = lambda x: x > 0.6  # Điều kiện với cosine similarity
    else:
        distances = torch.norm(embeddings_tensor - pred_embed, dim=1).detach().cpu().numpy()
        valid_condition = lambda x: x < 0.9  # Điều kiện với L2

    # Tìm `k` chỉ số gần nhất
    if distance_mode == 'l2':
        k_smallest_indices = np.argsort(distances)[:k]
    else:
        k_smallest_indices = np.argsort(-distances)[:k]  # Lớn hơn là gần hơn với cosine similarity

    # Kiểm tra điều kiện mẫu
    valid_distances = [distances[idx] for idx in k_smallest_indices if valid_condition(distances[idx])]

    if len(valid_distances) < k / 2:
        return -1  # Không đủ mẫu thỏa mãn điều kiện

    # Lấy các lớp tương ứng với `k` chỉ số gần nhất
    k_nearest_classes = [image2class[idx] for idx in k_smallest_indices]

    # Đếm số lần xuất hiện của các lớp
    class_counts = Counter(k_nearest_classes)

    # Lấy lớp xuất hiện nhiều nhất
    best_class_index, count = class_counts.most_common(1)[0]

    return k_nearest_classes
    # Kiểm tra ngưỡng đa số
    if count >= threshold * len(k_nearest_classes):
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
   
    image_path = 'testdata/sontung/005.jpg'
    image = Image.open(image_path).convert('RGB')
    align_image, faces, probs, lanmark  = get_align(image)
    pred_embed= infer(recogn_model, align_image)
    result = find_closest_person(pred_embed, embeddings, image2class, 'cosine')
    print(result)
    