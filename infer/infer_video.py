import torch
import cv2
import torch.nn.functional as F
from .infer_image import infer
from .get_embedding import load_embeddings_and_names
from .getface import yolo
from torch.nn.modules.distance import PairwiseDistance
from PIL import Image
from models.spoofing.FasNet import Fasnet
import numpy as np
from collections import Counter
from .getface import mtcnn_inceptionresnetV1
from models.face_detect.OpenCv import OpenCvClient
from .infer_image import infer, get_align
from .utils import get_model
from gtts import gTTS
import pygame
import os

l2_distance = PairwiseDistance(p=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
antispoof_model = Fasnet()
opencv = OpenCvClient()


def infer_camera(embeddings, names, recogn_model_name, distance_mode='cosine', min_face_area=40000, threshold=0.7, required_images=50):
 
   
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Không thể mở camera")
        return

    valid_images = []  # Danh sách lưu các input image hợp lệ
    
    # Các biến để theo dõi trạng thái trước đó
    previous_message = 0   # 0: don't have face, 1: detect face, 2: face is skewed, 3: face is too far away, 4: fake face or low confident

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể chụp được hình ảnh")
            break

        # Gọi hàm nhận diện khuôn mặt và chống giả mạo
        input_image, face, prob, landmark, is_real, antispoof_score = get_align(frame, antispoof_model)

        # Kiểm tra khuôn mặt và xử lý các trường hợp khác
        if face is not None:  # Nếu phát hiện khuôn mặt
            x1, y1, x2, y2 = map(int, face)
            if prob > threshold:  # Chỉ vẽ nếu confidence vượt ngưỡng
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Face {prob:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            area = (face[2] - face[0]) * (face[3] - face[1])
        
            if prob > threshold and is_real:
                # Tính trung tâm khuôn mặt
                center = np.mean(landmark, axis=0)
                height, width, _ = frame.shape
                center_x, center_y = center

                # Kiểm tra khuôn mặt ở giữa khung hình
                distance_from_center = np.sqrt((center_x - width / 2) ** 2 + (center_y - height / 2) ** 2)
                if area > min_face_area:

                    if width * 0.15 < center_x < width * 0.85 and height * 0.15 < center_y < height * 0.85 and distance_from_center < min(width, height) * 0.4:
                        if previous_message != 1:
                            print('Giữ khuôn mặt yên')
                            previous_message = 1
                        valid_images.append(input_image)

                    else:
                        if previous_message != 2:
                            print('Di chuyển khuôn mặt vào giữa khung hình')
                            previous_message = 2

                else:
                    if previous_message != 3:
                        print("Đưa khuôn mặt lại gần hơn")
                        previous_message = 3
        
            else:
                if previous_message != 4:
                    print('Khuôn mặt giả hoặc độ tin cậy quá thấp')
                    previous_message = 4

        else:
            if previous_message != 0:
                print("Không phát hiện khuôn mặt")
                previous_message = 0

        cv2.imshow('FACE RECOGNITON', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Dừng vòng lặp nếu đã thu thập đủ số ảnh hợp lệ
        if len(valid_images) >= required_images:
            print(f"Đã thu thập đủ {required_images} ảnh hợp lệ.")
            break
     
    # Giải phóng camera và đóng cửa sổ
    cap.release()
    cv2.destroyAllWindows()

    return valid_images



def display_images(images, delay=100):
    recogn_model = get_model('inceptionresnetV1')
    if not images:
        print("No images to display.")
        return

    for idx, image in enumerate(images):
        image = np.array(image.permute(1,2,0))
        if not isinstance(image, np.ndarray):
            print(f"Invalid image at index {idx}. Skipping...")
            continue
        # Hiển thị hình ảnh với tên cửa sổ là thứ tự của ảnh
        cv2.imshow(f'Image {idx + 1}', image)
        cv2.waitKey(delay)  # Đợi trong khoảng thời gian delay trước khi hiển thị ảnh tiếp theo
        cv2.destroyWindow(f'Image {idx + 1}')  # Đóng cửa sổ hiện tại

    cv2.destroyAllWindows()



if __name__ == '__main__':
 
    recogn_model_name = 'inceptionresnetV1'
    embedding_file_path = f'data/data_source/{recogn_model_name}_embeddings.npy'
    names_file_path = f'data/data_source/{recogn_model_name}_image2class.pkl'
   
    embeddings, names = load_embeddings_and_names(embedding_file_path, names_file_path)

    valid_image = infer_camera(embeddings, names, recogn_model_name, 'cosine')

    display_images(valid_image)
  

   

