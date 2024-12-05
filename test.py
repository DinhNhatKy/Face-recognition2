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
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns


workers = 0 if os.name == 'nt' else 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def collate_fn(x):
    return x[0]



def test_model(test_folder, recognition_model_name, embedding_file_path, image2class_file_path, index2class_file_path, batch_size=512):
    recognition_model = get_model(recognition_model_name)


    dataset = datasets.ImageFolder(test_folder)
    dataset.index2class = {i: c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)
    
    aligned = []  # List of aligned images in batches
    image2class_test = {}

    for i, (x, y) in enumerate(loader):
        image2class_test[i] = y
        x_aligned = mtcnn_inceptionresnetV1(x)
    
        if x_aligned is not None:
            x_aligned = inceptionresnetV1_transform(x_aligned)
            aligned.append(x_aligned)
        else:
            results = yolo(x)
            if results[0].boxes.xyxy.shape[0] != 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                x1, y1, x2, y2 = map(int, boxes[0])
                face = x.crop((x1, y1, x2, y2)).resize((160, 160), Image.Resampling.LANCZOS)
                x_aligned = inceptionresnetV1_transform(face)
                aligned.append(x_aligned)
            else:
                print('No detect face, get full image')
                face = x.resize((160, 160), Image.Resampling.LANCZOS)
                x_aligned = inceptionresnetV1_transform(face)
                aligned.append(x_aligned)

        # Process batch when it reaches the batch_size
        if len(aligned) >= batch_size:
            batch = torch.cat(aligned[:batch_size], dim=0).to(device)
            batch_embeddings = recognition_model(batch).detach().cpu()
            if 'test_embeddings' not in locals():
                test_embeddings = batch_embeddings
            else:
                test_embeddings = torch.cat((test_embeddings, batch_embeddings), dim=0)

            # Remove processed items
            aligned = aligned[batch_size:]

    # Process remaining items in the aligned list
    if aligned:
        batch = torch.cat(aligned, dim=0).to(device)
        batch_embeddings = recognition_model(batch).detach().cpu()
        if 'test_embeddings' not in locals():
            test_embeddings = batch_embeddings
        else:
            test_embeddings = torch.cat((test_embeddings, batch_embeddings), dim=0)

    embeddings, image2class, index2class = load_embeddings_and_names(
        embedding_file_path, image2class_file_path, index2class_file_path
    )

    class_ids = []
    for i, test_embedding in enumerate(test_embeddings):
        class_id = find_closest_person(
            test_embedding, 
            embeddings, 
            image2class
        )
        class_ids.append((class_id, image2class_test[i]))

    print(len(class_ids))
    print(class_ids)

    matching_elements = sum(1 for item in class_ids if item[0] == item[1])
    total_elements = len(class_ids)
    percentage = (matching_elements / total_elements) * 100

    return percentage


    def collate_fn(x):
        return x[0]
    
    
def test_model2(test_folder_1, test_folder_2, recognition_model_name, batch_size=512, device='cpu'):
    recognition_model = get_model(recognition_model_name).to(device)
    recognition_model.eval()  # Đảm bảo model ở chế độ evaluation



    # Đọc dữ liệu từ hai thư mục test
    dataset_1 = datasets.ImageFolder(test_folder_1)
    dataset_2 = datasets.ImageFolder(test_folder_2)

    dataset_1.index2class = {i: c for c, i in dataset_1.class_to_idx.items()}
    dataset_2.index2class = {i: c for c, i in dataset_2.class_to_idx.items()}

    loader_1 = DataLoader(dataset_1, collate_fn=collate_fn, batch_size=batch_size, num_workers=4)
    loader_2 = DataLoader(dataset_2, collate_fn=collate_fn, batch_size=batch_size, num_workers=4)

    aligned_1 = []  # List of aligned images from folder 1
    aligned_2 = []  # List of aligned images from folder 2

    # Lấy embeddings từ folder 1
    with torch.no_grad():  # Tắt tính toán gradient vì ta chỉ cần forward pass
        for x, _ in loader_1:
            x_aligned = mtcnn_inceptionresnetV1(x)
            x_aligned = x_aligned.unsqueeze(0).to(device)
            embeddings = recognition_model(x_aligned)
            aligned_1.append(embeddings)

    # Lấy embeddings từ folder 2
    with torch.no_grad():
        for x, _ in loader_2:
            x_aligned = mtcnn_inceptionresnetV1(x)
            x_aligned = x_aligned.unsqueeze(0).to(device)
            embeddings = recognition_model(x_aligned)
            aligned_2.append(embeddings)

    # Chuyển embeddings sang tensor
    embeddings_1 = torch.cat(aligned_1, dim=0).cpu().numpy()
    embeddings_2 = torch.cat(aligned_2, dim=0).cpu().numpy()

    # Tính cosine similarity giữa embeddings từ folder 1 và folder 2
    cosine_sim_matrix = cosine_similarity(embeddings_1, embeddings_2)

    # Vẽ ma trận tương quan
    plt.figure(figsize=(8, 6))
    sns.heatmap(cosine_sim_matrix, annot=True, cmap='coolwarm', fmt='.2f', xticklabels=[f'Image {i+1}' for i in range(len(embeddings_2))], yticklabels=[f'Image {i+1}' for i in range(len(embeddings_1))])
    plt.title('Cosine Similarity Matrix between two sets of images')
    plt.xlabel('Images in Folder 2')
    plt.ylabel('Images in Folder 1')
    plt.show()

    return cosine_sim_matrix


if __name__ == "__main__":
    # test_folder = 'data/data_gallery_1'

    # embedding_file_path= 'data/data_source/db1/inceptionresnetV1_embeddings.npy'
    # image2class_file_path = 'data/data_source/db1/inceptionresnetV1_image2class.pkl'
    # index2class_file_path = 'data/data_source/db1/inceptionresnetV1_index2class.pkl'

    # percentage = test_model(test_folder=test_folder, 
    #            recognition_model_name='inceptionresnetV1',
    #            embedding_file_path=embedding_file_path,
    #            image2class_file_path=image2class_file_path,
    #            index2class_file_path= index2class_file_path,
    #            )
    
    # print('accuracy: ', percentage)
    folder1 = 'data/folder1'
    folder2 = 'data/folder2'
    test_model2(folder1, folder2,  'inceptionresnetV1')


