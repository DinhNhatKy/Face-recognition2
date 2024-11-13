import torch
import matplotlib.pyplot as plt
import numpy as np
from .utils import get_model
from PIL import Image
import torch
from torchvision import transforms
from torch.nn.modules.distance import PairwiseDistance
from .getface import mtcnn_inceptionresnetV1, mtcnn_inceptionresnetV2, mtcnn_resnet, yolo
from models.inceptionresnetV2 import InceptionResnetV2Triplet
from models.resnet import Resnet34Triplet
from models.inceptionresnetV1 import InceptionResnetV1
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def inceptionresnetV1_transform(img):
    img = img.unsqueeze(0)
    img = img.to(device)
    return img


def resnet_transform(image, image_size):
    data_transforms = transforms.Compose([
    transforms.Resize(size= image_size), 
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.6071, 0.4609, 0.3944], 
        std=[0.2457, 0.2175, 0.2129]  
    )
    ])

    img = data_transforms(image)
    img = img.unsqueeze(0)
    img = img.to(device)

    return img

def inceptionresnetV2_transform(image, image_size):
    data_transforms = transforms.Compose([
    transforms.Resize(size= image_size), 
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5], 
        std=[0.5, 0.5, 0.5]  
    )
    ])
    img = data_transforms(image)
    img = img.unsqueeze(0)
    img = img.to(device)

    return img



def infer(recogn_model_name, image_path):
    image = Image.open(image_path).convert('RGB')

    recogn_model = get_model(recogn_model_name)

    if isinstance(recogn_model, Resnet34Triplet):
        input_image = mtcnn_resnet(image)
        input_image = resnet_transform(input_image, 140)

    elif isinstance(recogn_model, InceptionResnetV2Triplet) :
        input_image = mtcnn_inceptionresnetV2(image)
        input_image = inceptionresnetV2_transform(input_image, (299, 299))

    elif isinstance(recogn_model, InceptionResnetV1):
        input_image = mtcnn_inceptionresnetV1(image)
        input_image = inceptionresnetV1_transform(input_image)

    else:
        print('No model !!')

    embedding = recogn_model(input_image)
    return embedding


if __name__ == "__main__":

    anc_path  = 'data/dataset/sontung/007.jpg'
    pos_path = 'data/dataset/sontung/012.jpg'
    neg_path = 'data/dataset/khanh/003.jpg'
    
    select_model = 'inceptionresnetV1'
    anc_embedding = infer(select_model , anc_path)
    pos_embedding =  infer(select_model , pos_path)
    neg_embedding =  infer(select_model , neg_path)

    l2_distance = PairwiseDistance(p=2)

    dist1 =  l2_distance.forward(anc_embedding, pos_embedding)
    dist2 =  l2_distance.forward(anc_embedding, neg_embedding)


    cosine_similarity = F.cosine_similarity

    similarity_pos = cosine_similarity(anc_embedding, pos_embedding, dim=1)
    similarity_neg = cosine_similarity(anc_embedding, neg_embedding, dim=1)

    print('l2:')
    print(dist1.item())
    print(dist2.item())
    print('cosine:')
    print(similarity_pos.item())
    print(similarity_neg.item())

