import torch
import matplotlib.pyplot as plt
import numpy as np
from .utils import get_model
from PIL import Image
import torch
from torchvision import transforms
from torch.nn.modules.distance import PairwiseDistance
from .getface import mtcnn_inceptionresnetV1, mtcnn_resnet, yolo
from models.resnet import Resnet34Triplet
from models.inceptionresnetV1 import InceptionResnetV1
import torch.nn.functional as F
from deepface.models.spoofing import FasNet



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def inceptionresnetV1_transform(img):
    if not isinstance(img, torch.Tensor):
        img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    return img

def resnet_transform(image):
    data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size=140),
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


def infer(recogn_model_name, image, antispoof_model):
    recogn_model = get_model(recogn_model_name)
    cv_image = np.array(image)

    faces, probs = mtcnn_inceptionresnetV1.detect(image)
    
    if faces is not None and len(faces) > 0:
        max_face = max(faces, key=lambda face: (face[2] - face[0]) * (face[3] - face[1]))

        is_real, antispoof_score = antispoof_model.analyze(img=cv_image, facial_area=tuple(max_face))

        input_image = mtcnn_inceptionresnetV1(image)
        input_image = inceptionresnetV1_transform(input_image)

        print(f'Antispoof score: {antispoof_score}, Is real: {is_real}')

    else:
        print("No faces detected!")
        return None, None
    
    embedding = recogn_model(input_image)
    return embedding, is_real




if __name__ == "__main__":
    
    antispoof_model =FasNet.Fasnet()

    anc_path  = 'testdata/chipu/002.jpg'
    pos_path = 'data/dataset/chipu/004.jpg'
    neg_path = 'data/dataset/sontung/001.jpg'
    
    select_model = 'inceptionresnetV1'

    anc_image = Image.open(anc_path).convert('RGB')
    pos_image = Image.open(pos_path).convert('RGB')
    neg_image = Image.open(neg_path).convert('RGB')

    anc_embedding, anc_spool_score= infer(select_model , anc_image, antispoof_model)
    pos_embedding, pos_spool_score =  infer(select_model , pos_image, antispoof_model)
    neg_embedding, neg_spool_score=  infer(select_model , neg_image, antispoof_model)

    print(anc_spool_score)
    print(pos_spool_score)
    print(neg_spool_score)

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




