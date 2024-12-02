from infer.infer_video import check_validation
from infer.infer_image import get_align, infer
from infer.identity_person import find_closest_person, find_closest_person_vote
from infer.get_embedding import load_embeddings_and_names
from torchvision import datasets
from infer.utils import get_model
from torch.utils.data import DataLoader
from infer.getface import mtcnn_inceptionresnetV1
from infer.getface import yolo
from infer.infer_image import inceptionresnetV1_transform
import os
from PIL import Image
import torch


workers = 0 if os.name == 'nt' else 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_model(test_folder, recognition_model_name, embedding_file_path, image2class_file_path, index2class_file_path):
    recognition_model = get_model(recognition_model_name)

    def collate_fn(x):
        return x[0]

    dataset = datasets.ImageFolder(test_folder)
    dataset.index2class = {i: c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)
    aligned = []  # List of images in the gallery

    image2class_test = {}
    for i, (x, y) in enumerate(loader):
        image2class_test[i] = y
        x_aligned = mtcnn_inceptionresnetV1(x)
    
        if x_aligned is not None:
            x_aligned = inceptionresnetV1_transform(x_aligned)
            aligned.append(x_aligned)
        else:
            results = yolo(x)
            print()
            if results[0].boxes.xyxy.shape[0] != 0:
                boxes = results[0].boxes.xyxy.cpu().numpy() 
                x1, y1, x2, y2 = map(int, boxes[0]) 
                face = x.crop((x1, y1, x2, y2)).resize((160, 160), Image.Resampling.LANCZOS)
                x_aligned = inceptionresnetV1_transform(face)
                aligned.append(x_aligned)
            else:
                face = x.resize((160, 160), Image.Resampling.LANCZOS)
                x_aligned = inceptionresnetV1_transform(face)
                aligned.append(x_aligned)

    if aligned:
        aligned = torch.cat(aligned, dim=0).to(device)
        test_embeddings = recognition_model(aligned).detach().cpu() 

    embeddings, image2class, index2class = load_embeddings_and_names(embedding_file_path, image2class_file_path, index2class_file_path)
    class_ids = []
    for i, test_embedding in enumerate(test_embeddings):
        class_id = find_closest_person(
        test_embedding, 
        embeddings, 
        image2class)
        class_ids.append((class_id, image2class_test[i]))
        
    print(len(class_ids))
    print(class_ids)

    matching_elements = sum(1 for item in class_ids if item[0] == item[1])

    total_elements = len(class_ids)
    percentage = (matching_elements / total_elements) * 100

    return percentage

if __name__ == "__main__":
    test_folder = 'data/data_gallery_1'

    embedding_file_path= 'data/data_source/db1/inceptionresnetV1_embeddings.npy'
    image2class_file_path = 'data/data_source/db1/inceptionresnetV1_image2class.pkl'
    index2class_file_path = 'data/data_source/db1/inceptionresnetV1_index2class.pkl'

    percentage = test_model(test_folder=test_folder, 
               recognition_model_name='inceptionresnetV1',
               embedding_file_path=embedding_file_path,
               image2class_file_path=image2class_file_path,
               index2class_file_path= index2class_file_path,
               )
    
    print(percentage)