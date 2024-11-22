import streamlit as st
import cv2
import numpy as np
from gtts import gTTS
import os
from collections import Counter
from infer.infer_video import infer_video, infer_camera, check_validation
from infer.infer_image import infer, get_align
from infer.get_embedding import load_embeddings_and_names
from infer.identity_person import find_closest_person_vote, find_closest_person
from infer.utils import get_model
import tempfile
import time

def main():
    # Load embeddings and names
    recogn_model_name = 'inceptionresnetV1'
    embedding_file_path = f'data/data_source/{recogn_model_name}_embeddings.npy'
    image2class_file_path = f'data/data_source/{recogn_model_name}_image2class.pkl'
    index2class_file_path = f'data/data_source/{recogn_model_name}_index2class.pkl'
    embeddings, image2class, index2class = load_embeddings_and_names(embedding_file_path, image2class_file_path, index2class_file_path)
    recogn_model = get_model('inceptionresnetV1')


    st.title("Ứng Dụng Nhận Diện Khuôn Mặt")
    st.write("Chọn một trong hai lựa chọn sau để thực hiện nhận diện khuôn mặt:")

    # Sidebar for parameter settings
    st.sidebar.title("Thiết lập siêu tham số")
    min_face_area = st.sidebar.slider("Kích thước tối thiểu khuôn mặt", 5000, 50000, 10000)
    bbox_threshold = st.sidebar.slider("Ngưỡng phát hiện khuôn mặt", 0.1, 1.0, 0.7)
    required_images = st.sidebar.slider("Số lượng ảnh cần thiết", 1, 50, 16)
    validation_threshold = st.sidebar.slider("Ngưỡng xác thực", 0.1, 1.0, 0.7)
    is_antispoof = st.sidebar.checkbox("Bật kiểm tra chống giả mạo", True)
    is_vote = st.sidebar.checkbox("Sử dụng phương pháp biểu quyết", False)
    distance_mode = st.sidebar.selectbox("Phương pháp tính khoảng cách", ["cosine", "l2"])

    if st.button('Nhận diện khuôn mặt qua Camera'):
        st.write("Đang mở camera...")
        # Kết nối camera
        camera = cv2.VideoCapture(0)  # `0` là camera mặc định

        if not camera.isOpened():
            st.error("Không thể mở camera!")
        else:
            # Hiển thị khung hình
            frame_placeholder = st.empty()  # Vùng trống để hiển thị khung hình
            
            try:
                while True:
                    ret, frame = camera.read()
                    if not ret:
                        st.error("Không thể đọc dữ liệu từ camera!")
                        break
                    
                    # # Thực hiện xử lý nhận diện khuôn mặt trên frame
                    # processed_frame = infer_camera(
                    #     frame,
                    #     min_face_area=min_face_area,
                    #     bbox_threshold=bbox_threshold,
                    #     required_images=required_images
                    # )
                    
                    # # Chuyển đổi frame sang định dạng hiển thị
                    # rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    # frame_placeholder.image(rgb_frame, channels="RGB")

            except Exception as e:
                st.error(f"Lỗi khi xử lý camera: {e}")
            finally:
                camera.release()  # Giải phóng camera
                st.write("Camera đã đóng.")


    uploaded_video = st.file_uploader("Tải video lên", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        st.write("Đang xử lý video...")

        # Tạo tệp tạm thời để lưu video tải lên
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        video_path = tfile.name  # Đường dẫn tệp video tạm thời

        try:
            result = infer_video(
                video_path,
                min_face_area=min_face_area,
                bbox_threshold=bbox_threshold,
                required_images=required_images
            )
            check_validation(
                result,
                embeddings,
                image2class,
                index2class,
                recogn_model,
                is_antispoof=is_antispoof,
                validation_threhold=validation_threshold,
                is_vote=is_vote,
                distance_mode=distance_mode
            )
        finally:
            # Đảm bảo video được giải phóng và tệp có thể bị xóa
            time.sleep(1)  # Thêm thời gian để hệ điều hành giải phóng tệp

            # Xóa tệp video tạm thời sau khi xử lý xong
            try:
                os.remove(video_path)
                st.write(f"Tệp video {video_path} đã được xóa.")
            except Exception as e:
                st.error(f"Không thể xóa tệp: {e}")

if __name__ == '__main__':
    main()
