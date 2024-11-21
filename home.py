import streamlit as st
import cv2
import numpy as np
from gtts import gTTS
import os
from collections import Counter
from infer.infer_video import infer_video, infer_camera, check_validation
from infer.infer_image import infer, get_align
from infer.get_embedding import load_embeddings_and_names
from infer.identity_person import find_closest_person_vote
from infer.getface import mtcnn_inceptionresnetV1
from models.spoofing.FasNet import Fasnet
import tempfile
import time


# Load embeddings and names
recogn_model_name = 'inceptionresnetV1'
embedding_file_path = f'data/data_source/{recogn_model_name}_embeddings.npy'
image2class_file_path = f'data/data_source/{recogn_model_name}_image2class.pkl'
index2class_file_path = f'data/data_source/{recogn_model_name}_index2class.pkl'
embeddings, image2class, index2class = load_embeddings_and_names(embedding_file_path, image2class_file_path, index2class_file_path)


st.title("Ứng Dụng Nhận Diện Khuôn Mặt")
st.write("Chọn một trong hai lựa chọn sau để thực hiện nhận diện khuôn mặt:")

if st.button('Nhận diện khuôn mặt qua Camera'):
    st.write("Đang xử lý nhận diện qua Camera...")
    valid_images = infer_camera(embeddings, image2class, recogn_model_name)
    check_validation(valid_images, embeddings, image2class, index2class)

uploaded_video = st.file_uploader("Tải video lên", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    st.write("Đang xử lý video...")
    
    # Tạo tệp tạm thời để lưu video tải lên
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_video.read())
    video_path = tfile.name  # Đường dẫn tệp video tạm thời

    try:
        # Giả sử infer_video là một hàm bạn đã định nghĩa để xử lý video
        valid_images = infer_video(video_path)
        check_validation(valid_images, embeddings, image2class, index2class)
    finally:
        # Đảm bảo video được giải phóng và tệp có thể bị xóa
        time.sleep(1)  # Thêm thời gian để hệ điều hành giải phóng tệp

        # Xóa tệp video tạm thời sau khi xử lý xong
        try:
            os.remove(video_path)
            st.write(f"Tệp video {video_path} đã được xóa.")
        except Exception as e:
            st.error(f"Không thể xóa tệp: {e}")