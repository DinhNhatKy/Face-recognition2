from torchvision import datasets
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import os
from .getface import get_face_mtcnn, get_face_yolo, yolo,  mtcnn
from .infer_image import get_model, infer
from torch.nn.modules.distance import PairwiseDistance
import pickle
import cv2
from PIL import Image
from supervision import Detections
from .get_embedding import load_embeddings_and_names

l2_distance = PairwiseDistance(p=2)
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')


def find_closest_person(pred_path, embeddings_path, names_path, detect_model_name, recogn_model_name):
 
    pred_embed, boxes = infer(detect_model_name, recogn_model_name, pred_path)

    embeddings, names = load_embeddings_and_names(embeddings_path, names_path)

    if isinstance(pred_embed, torch.Tensor):
        pred_embed = pred_embed.cpu()  

    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)

    scores = l2_distance(pred_embed.unsqueeze(0), embeddings_tensor)
    scores = scores.detach().cpu().numpy()

    optim_index = np.argmin(scores)
    name_of_person = names[optim_index]

    img = cv2.imread(pred_path)

    if boxes is not None:
        x1, y1, x2, y2 = map(int, boxes[0])
        cv2.rectangle(img, (x1, y1), (x2 , y2), (255, 0, 0), 1)
        cv2.putText(img, name_of_person, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Face Recognition', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return scores, optim_index, name_of_person


if __name__ == '__main__':

    recogn_model_name= 'inceptionresnetV1'
    test_folder_path = 'testdata/chipu'
    detect_model_name = 'yolo'
    embeddings_path = f'data/embedding_names/{recogn_model_name}_mtcnn_embeddings.npy'
    names_path = f'data/embedding_names/{recogn_model_name}_names.pkl'
   
   

    for image_name in os.listdir(test_folder_path):
        pred_path = os.path.join(test_folder_path, image_name)
        scorres, optim_index, name_of_person = find_closest_person(pred_path, embeddings_path, names_path, detect_model_name, recogn_model_name)
        print(optim_index)
        print(scorres.shape)
        print(name_of_person)

