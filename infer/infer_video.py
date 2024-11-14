import torch
import cv2
import torch.nn.functional as F
from .infer_image import infer
from .get_embedding import load_embeddings_and_names
from .getface import yolo
from torch.nn.modules.distance import PairwiseDistance

# Khởi tạo biến
l2_distance = PairwiseDistance(p=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def find_closest_person(pred_embed, embeddings, names, distance_mode):
    if isinstance(pred_embed, torch.Tensor):
        pred_embed = pred_embed.cpu()

    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)

    total_distances = {}
    counts = {}
    
    if distance_mode == 'cosine':
        distances = F.cosine_similarity(pred_embed, embeddings_tensor)
    else:
        distances = torch.norm(embeddings_tensor - pred_embed, dim=1).detach().cpu().numpy()

    for i, name in enumerate(names):
        distance = distances[i].item()  # Chuyển đổi tensor thành số
        if name not in total_distances:
            total_distances[name] = 0
            counts[name] = 0
        total_distances[name] += distance
        counts[name] += 1

    avg_distances = {name: total_distances[name] / counts[name] for name in total_distances}

    if distance_mode == 'l2':
        name_of_person = min(avg_distances, key=avg_distances.get)
    else:
        name_of_person = max(avg_distances, key=avg_distances.get)

    return avg_distances, name_of_person

def infer_video(video_path, embeddings, names, recogn_model_name, distance_mode):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pred_embed = infer(recogn_model_name, img)

        avg_distances, name_of_person = find_closest_person(pred_embed, embeddings, names, distance_mode)

        results = yolo(frame)
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, name_of_person, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Ví dụ gọi hàm
    video_path = 'testdata/sontung_video2.mp4'
    recogn_model_name = 'inceptionresnetV1'
    embedding_file_path = f'data/embedding_names/{recogn_model_name}_embeddings.npy'
    names_file_path = f'data/embedding_names/{recogn_model_name}_names.pkl'
   
    embeddings, names = load_embeddings_and_names(embedding_file_path, names_file_path)

    infer_video(video_path, embeddings, names, recogn_model_name, 'l2')