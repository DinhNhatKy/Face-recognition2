from torchvision import datasets
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import os
from .getface import mtcnn_inceptionresnetV1, mtcnn_resnet
from .infer_image import get_model, resnet_transform, inceptionresnetV1_transform
from torch.nn.modules.distance import PairwiseDistance
import pickle
from torchvision import transforms

from deepface.modules.detection import extract_faces
from deepface.modules.preprocessing import resize_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

workers = 0 if os.name == 'nt' else 4


def create_data_embeddings(data_gallary_path, recognition_model_name, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    recognition_model = get_model(recognition_model_name)

    def collate_fn(x):
        return x[0]

    dataset = datasets.ImageFolder(data_gallary_path)
    dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

    aligned = []  # List of images in the gallery
    names = []    # List of names corresponding to images


    for x, y in loader:
        x_aligned= mtcnn_inceptionresnetV1(x)

        if x_aligned is not None:
            x_aligned = inceptionresnetV1_transform(x_aligned)
            aligned.append(x_aligned)
            names.append(dataset.idx_to_class[y])


    if aligned:
        aligned = torch.cat(aligned, dim=0).to(device)
        embeddings = recognition_model(aligned).detach().cpu().numpy() 
      
        # Save embedding 
        embedding_file_path = os.path.join(save_path, f"{recognition_model_name}_embeddings.npy")
        np.save(embedding_file_path, embeddings)

        names_file_path = os.path.join(save_path, f"{recognition_model_name}_names.pkl")
        with open(names_file_path, 'wb') as f:
            pickle.dump(names, f)

        print(f"Embeddings saved to {embedding_file_path}")
        print(f"Names saved to {names_file_path}")
        
        return embeddings, names
    else:
        print("No aligned images found.")
       
def load_embeddings_and_names(embedding_file_path, names_file_path):
    
    embeddings = np.load(embedding_file_path)
    with open(names_file_path, 'rb') as f:
        names = pickle.load(f)

    return embeddings, names

if __name__ == '__main__':
    
    data_gallary_path = 'data/dataset'
    embedding_save_path = 'data/embedding_names'
    embeddings, names = create_data_embeddings(data_gallary_path, 'inceptionresnetV1', embedding_save_path )
 

    # embedding_file_path= 'data/embedding_names/inceptionresnetV1_embeddings.npy'
    # names_file_path = 'data/embedding_names/inceptionresnetV1_names.pkl'

    # embeddings, names = load_embeddings_and_names(embedding_file_path, names_file_path)

    
    print(embeddings.shape)
    print(names)
