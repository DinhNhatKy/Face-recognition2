import torch
import torch.nn as nn
from models.face_recogn.inceptionresnetV1 import InceptionResnetV1
from models.face_recogn.resnet import Resnet34Triplet
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_model(model_name):
    model = None

    if model_name == 'resnet34':
        checkpoint = torch.load('models/pretrained/model_resnet34_triplet.pt', weights_only = False, map_location=device)
        model = Resnet34Triplet(
            embedding_dimension=checkpoint['embedding_dimension'],
            pretrained=True
        )
        state_dict = checkpoint['model_state_dict']

    elif model_name == 'inceptionresnetV1':
        model = InceptionResnetV1(pretrained='vggface2', classify=False, num_classes=None, dropout_prob=0.6, device=device)
        state_dict = None

    else:
        print('please enter correct model! ')

    model, _ = set_model_gpu_mode(model)

    if state_dict:
        model.load_state_dict(state_dict)
    model.eval()
    
    return model



def set_model_gpu_mode(model):
    flag_train_gpu = torch.cuda.is_available()
    flag_train_multi_gpu = False

    if flag_train_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model.cuda()
        flag_train_multi_gpu = True
        print('Using multi-gpu training.')

    elif flag_train_gpu and torch.cuda.device_count() == 1:
        model.cuda()
        print('Using single-gpu training.')

    return model, flag_train_multi_gpu
