import torch
import matplotlib.pyplot as plt
import numpy as np
from .utils import get_model
from PIL import Image
import torch
from torchvision import transforms
from torch.nn.modules.distance import PairwiseDistance
from .getface import mtcnn_inceptionresnetV1, mtcnn_resnet, yolo
from models.face_recogn.resnet import Resnet34Triplet
from models.face_recogn.inceptionresnetV1 import InceptionResnetV1
import torch.nn.functional as F
from models.spoofing.FasNet import Fasnet
import cv2


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


def infer(recogn_model, align_image):
    try:
        input_image = inceptionresnetV1_transform(align_image)
        embedding = recogn_model(input_image)
        return embedding

    except Exception as e:
        print(f"Error during inference: {e}")
        return None
    

def get_align(image):
    face= None
    input_image = image
    prob = 0
    lanmark = None

    faces, probs, lanmarks = mtcnn_inceptionresnetV1.detect(image, landmarks= True)

    if faces is not None and len(faces) > 0:
        face = faces[0] # get highest area bboxes
        prob = probs[0]
        lanmark= lanmarks[0]
        input_image = mtcnn_inceptionresnetV1(image)

    return input_image, face, prob, lanmark


if __name__ == "__main__":
    
    antispoof_model = Fasnet()

    # anc_path  = 'testdata/chipu/002.jpg'
    # pos_path = 'data/dataset/chipu/004.jpg'
    # neg_path = 'data/dataset/sontung/001.jpg'
    
    # select_model = 'inceptionresnetV1'

    # anc_image = Image.open(anc_path).convert('RGB')
    # pos_image = Image.open(pos_path).convert('RGB')
    # neg_image = Image.open(neg_path).convert('RGB')

    # anc_embedding, anc_spool_score, anc_score= infer(select_model , anc_image, antispoof_model)
    # pos_embedding, pos_spool_score, pos_score =  infer(select_model , pos_image, antispoof_model)
    # neg_embedding, neg_spool_score, neg_score =  infer(select_model , neg_image, antispoof_model)

    # print(anc_spool_score, anc_score)
    # print(pos_spool_score, pos_score)
    # print(neg_spool_score, neg_score)

    # l2_distance = PairwiseDistance(p=2)

    # dist1 =  l2_distance.forward(anc_embedding, pos_embedding)
    # dist2 =  l2_distance.forward(anc_embedding, neg_embedding)


    # cosine_similarity = F.cosine_similarity

    # similarity_pos = cosine_similarity(anc_embedding, pos_embedding, dim=1)
    # similarity_neg = cosine_similarity(anc_embedding, neg_embedding, dim=1)

    # print('l2:')
    # print(dist1.item())
    # print(dist2.item())
    # print('cosine:')
    # print(similarity_pos.item())
    # print(similarity_neg.item())

    image = Image.open('testdata/thaotam/006.jpg').convert('RGB')
    input_image, face, prob, landmark = get_align(image)

    # In các thông tin nhận diện
    print(input_image.shape)
    print(face)
    print(prob)
    print(landmark)

    # Chuyển ảnh sang dạng numpy array và đổi màu từ RGB sang BGR cho OpenCV
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Vẽ hình chữ nhật bao quanh khuôn mặt
    x1, y1, x2, y2 = map(int, face)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Hiển thị xác suất nhận diện khuôn mặt
    cv2.putText(image, f"Face {prob:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Vẽ các điểm landmark (các chấm)
    for (x, y) in landmark:
        cv2.circle(image, (int(x), int(y)), 2, (0, 0, 255), -1)  # Chấm màu đỏ, bán kính 2


    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
