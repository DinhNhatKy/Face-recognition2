import torch
from facenet_pytorch import InceptionResnetV1, MTCNN, fixed_image_standardization
import matplotlib.pyplot as plt
import numpy as np
from .utils import set_model_architecture, set_model_gpu_mode
from PIL import Image
import torch
from torchvision import transforms
from torch.nn.modules.distance import PairwiseDistance
from .getface import get_face_mtcnn, get_face_yolo, yolo,  mtcnn
from models.inceptionresnetV2 import InceptionResnetV2Triplet
from models.resnet import Resnet34Triplet



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def get_model(model_name):
    if model_name == 'resnet34':
        checkpoint = torch.load('pretrained/model_resnet34_triplet.pt', weights_only = False, map_location=device)
        state_dict = checkpoint['model_state_dict']

    elif model_name == 'inceptionresnetV1':
        state_dict = None

    elif model_name == 'inceptionresnetV2':
        checkpoint = None # Haven't trained yet
        state_dict = None
    else:
        print('please enter correct model! ')

    model =  set_model_architecture(model_architecture= model_name, pretrained= True, embedding_dimension= 512)
    model, _ = set_model_gpu_mode(model)

    if state_dict:
        model.load_state_dict(state_dict)
    model.eval()
    
    return model


def transform_image(image, image_size):
    data_transforms = transforms.Compose([
    transforms.ToPILImage(),
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


def infer(detect_model_name, recogn_model_name, image_path):
    input_image = None
    if detect_model_name == 'yolo':
        input_image, boxes = get_face_yolo(image_path, yolo)
    elif detect_model_name=='mtcnn':
        input_image, boxes = get_face_mtcnn(image_path, mtcnn)
    else:
        print('please select correct detect model')
    
    recogn_model = get_model(recogn_model_name)


    if isinstance(recogn_model, Resnet34Triplet):
        input_image = resnet_transform(input_image, 140)
    elif isinstance(recogn_model, InceptionResnetV2Triplet) :
        input_image = inceptionresnetV2_transform(input_image, (299, 299))
    else:
        if input_image.size[0]< 80 or input_image.size[1]< 1:
            input_image =   input_image.resize((299, 299))
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)

    embedding = recogn_model(input_image)
    return embedding, boxes


if __name__ == "__main__":

    anc_path  = 'data/dataset/thaotam/018.jpg'
    pos_path = 'data/dataset/thaotam/017.jpg'
    neg_path = 'data/dataset/sontung/003.jpg'
    
    anc_embedding, _ = infer('mtcnn', 'inceptionresnetV1', anc_path)
    pos_embedding, _ = infer('mtcnn', 'inceptionresnetV1', pos_path)
    neg_embedding,_ = infer('mtcnn', 'inceptionresnetV1', neg_path)

    l2_distance = PairwiseDistance(p=2)
    dist1 =  l2_distance.forward(anc_embedding, pos_embedding)
    dist2 =  l2_distance.forward(anc_embedding, neg_embedding)

    print(dist1.item())
    print(dist2.item())
