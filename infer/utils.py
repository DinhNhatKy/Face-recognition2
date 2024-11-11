import torch.optim as optim
import torch
import torch.nn as nn
from models.inceptionresnetV2 import InceptionResnetV2Triplet
from models.inceptionresnetV1 import InceptionResnetV1
from models.resnet import Resnet34Triplet
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_model_architecture(model_architecture, pretrained, embedding_dimension):

    if model_architecture == "resnet34":
        model = Resnet34Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "inceptionresnetV2":
        model = InceptionResnetV2Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == 'inceptionresnetV1':
        model = InceptionResnetV1(pretrained='vggface2', classify=False, num_classes=None, dropout_prob=0.6, device=device)
    else:
        print('please select correct model')
        
    print("Using {} model architecture.".format(model_architecture))

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
