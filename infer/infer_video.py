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
import os
from gtts import gTTS
from .identity_person import find_closest_person_vote,find_closest_person

recogn_model = get_model('inceptionresnetV1')
l2_distance = PairwiseDistance(p=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
antispoof_model = Fasnet()
opencv = OpenCvClient()


def infer_camera(min_face_area=10000, bbox_threshold=0.7, required_images=14):
   
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
        input_image, face, prob, landmark = get_align(frame)

        # Check for face and handle different conditions
        if face is not None:  # If a face is detected
            x1, y1, x2, y2 = map(int, face)
            if prob > bbox_threshold:  # Only draw if the confidence is above the threshold
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Face {prob:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            area = (face[2] - face[0]) * (face[3] - face[1])

            if prob > bbox_threshold:
                # Calculate the center of the face
                center = np.mean(landmark, axis=0)
                height, width, _ = frame.shape
                center_x, center_y = center

                # Check if the face is centered in the frame
                distance_from_center = np.sqrt((center_x - width / 2) ** 2 + (center_y - height / 2) ** 2)
                if area > min_face_area:

                    if width * 0.15 < center_x < width * 0.85 and height * 0.15 < center_y < height * 0.85 and distance_from_center < min(width, height) * 0.4:
                        if previous_message != 1:
                            tts = gTTS("giữ yên khuôn mặt", lang='vi')
                            tts.save("guide.mp3") 
                            os.system("start guide.mp3") 

                            previous_message = 1
                        valid_images.append(input_image)

                    else:
                        if previous_message != 2:
                            tts = gTTS("đưa khuôn mặt vào giữa màn hình",  lang='vi')
                            tts.save("guide.mp3") 
                            os.system("start guide.mp3") 

                            previous_message = 2

                else:
                    if previous_message != 3:
                        tts = gTTS("Đưa khuôn mặt lại gần hơn",  lang='vi')
                        previous_message = 3

        else:
            if previous_message != 0:
                print("Không phát hiện khuôn mặt")
                

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



def infer_video(video_path, min_face_area=10000, bbox_threshold=0.7, required_images= 14):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Không thể mở video")
        return

    valid_images = []  # List to store valid input images

    # Variables to track previous states
    previous_message = 0   # 0: no face, 1: face detected, 2: face is skewed, 3: face is too far, 4: fake face

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc video")
            break

        # Call the face alignment and anti-spoofing function
        input_image, face, prob, landmark = get_align(frame)
     
        # Check for face and handle different conditions
        if face is not None:  # If a face is detected
            x1, y1, x2, y2 = map(int, face)
            if prob > bbox_threshold:  # Only draw if the confidence is above the threshold
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Face {prob:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            area = (face[2] - face[0]) * (face[3] - face[1])

            if prob > bbox_threshold:
                # Calculate the center of the face
                center = np.mean(landmark, axis=0)
                height, width, _ = frame.shape
                center_x, center_y = center

                # Check if the face is centered in the frame
                distance_from_center = np.sqrt((center_x - width / 2) ** 2 + (center_y - height / 2) ** 2)
                if area > min_face_area:

                    if width * 0.15 < center_x < width * 0.85 and height * 0.15 < center_y < height * 0.85 and distance_from_center < min(width, height) * 0.4:
                        if previous_message != 1:
                            tts = gTTS("giữ yên khuôn mặt")
                            tts.save("guide.mp3") 
                            os.system("start guide.mp3") 

                            previous_message = 1
                        valid_images.append(input_image)

                    else:
                        if previous_message != 2:
                            tts = gTTS("đưa khuôn mặt vào giữa màn hình")
                            tts.save("guide.mp3") 
                            os.system("start guide.mp3") 

                            previous_message = 2

                else:
                    if previous_message != 3:
                        tts = gTTS("Đưa khuôn mặt lại gần hơn")
                        previous_message = 3

        else:
            if previous_message != 0:
                print("Không phát hiện khuôn mặt")
                

        # Display the frame
        cv2.imshow('FACE RECOGNITION', frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Stop the loop if enough valid images are collected
        if len(valid_images) >= required_images:
            print(f"Đã thu thập đủ {required_images} ảnh hợp lệ.")
            break

    # Release the video capture object and close any open windows
    cap.release()
    cv2.destroyAllWindows()

    return valid_images



def check_validation(images, embeddings, image2class, idx_to_class, recogn_model, is_antispoof= False):
    


    if len(images) == 0:
        print("Không có ảnh để xử lý.")
        return
    
    predict_class = []

    for image in images:
        # if is_antispoof:
        #     antispoof_model = Fasnet()
        #     is_real, score = antispoof_model.analyze(image, (0, 0 , image.shape[1], image.shape[0]))
        #     if not is_real and score > 0.7:
        #         continue
        pred_embed = infer(recogn_model, image)
        result = find_closest_person_vote(pred_embed, embeddings, image2class)
        if result != -1:
            predict_class.append(result)

    class_count = Counter(predict_class)
    
    
    # Tính ngưỡng đa số (lớn hơn 75% số dự đoán)
    majority_threshold = len(images) * 0.75

    person_identified = False  # Biến để theo dõi xem có tìm được người nhận diện hay không

    # Kiểm tra xem có class nào xuất hiện quá 75% số lần không
    for cls, count in class_count.items():
        if count >= majority_threshold:
            person_name = idx_to_class.get(cls, 'Unknown')
            print(f"Người được nhận diện là: {person_name}")
            
            # Tạo giọng nói bằng gTTS
            tts = gTTS(f"Xin chào {person_name}", lang='vi')
            tts.save("greeting.mp3")  # Lưu file âm thanh tạm thời
            os.system("start greeting.mp3")  # Chạy âm thanh (Windows)
            
            person_identified = True
            break  # Dừng lại sau khi tìm được người nhận diện đầu tiên thỏa mãn điều kiện
    
    if not person_identified:
        print("Unknown person")
        tts = gTTS("Vui lòng thử lại", lang='vi')
        tts.save("retry.mp3") 
        os.system("start retry.mp3") 


if __name__ == '__main__':

    recogn_model_name = 'inceptionresnetV1'
    embedding_file_path = f'data/data_source/{recogn_model_name}_embeddings.npy'
    image2class_file_path = f'data/data_source/{recogn_model_name}_image2class.pkl'
    index2class_file_path = f'data/data_source/{recogn_model_name}_index2class.pkl'
    
    embeddings, image2class, index2class = load_embeddings_and_names(embedding_file_path, image2class_file_path, index2class_file_path)
    video_path = 'data/dataset/sontung/025.jpg'
    valid_images = infer_camera()

    check_validation(valid_images, embeddings, image2class, index2class, recogn_model, is_antispoof= True)



