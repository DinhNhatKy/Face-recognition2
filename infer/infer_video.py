import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
import numpy as np
from PIL import Image
import cv2  # Để xử lý bbox
from huggingface_hub import hf_hub_download
from ultralytics import YOLO


resnet = InceptionResnetV1(
    classify=True,
    pretrained='vggface2',
    num_classes=len(dataset.class_to_idx)
).to(device)


model_path = '/kaggle/working/model_final.pth'
checkpoint = torch.load(model_path)
resnet.load_state_dict(checkpoint['model_state_dict'])


model_path_yolo = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
yolo_model = YOLO(model_path_yolo)

def infer_video(model, video_path, output_path):
    # Mở video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video file")
        return

    # Lấy thông tin về kích thước video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Tạo đối tượng VideoWriter để lưu video đầu ra
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

    while True:
        # Đọc từng khung hình từ video
        ret, frame = cap.read()
        if not ret:
            break  # Kết thúc vòng lặp nếu không còn khung hình

        # Chuyển đổi khung hình sang RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)

        # Phát hiện khuôn mặt và căn chỉnh
        x_aligned, prob = mtcnn(pil_image, return_prob=True)

        if x_aligned is None:
            print("Can't find face in frame")
            continue

        x_aligned = x_aligned.unsqueeze(0).to(device)

        # Dự đoán lớp
        resnet.eval()
        with torch.no_grad():
            output = model(x_aligned)

        predicted_class = torch.argmax(output, dim=1)  # Lớp được dự đoán
        person_name = dataset.idx_to_class[predicted_class.item()]

        # Phát hiện khuôn mặt với YOLO
        results = yolo_model(pil_image)
        boxes = results[0].boxes.xyxy  # Lấy bounding box
        confidences = results[0].boxes.conf  # Lấy độ tin cậy

        # Vẽ bounding box lên ảnh và thêm tên người
        for box, conf in zip(boxes, confidences):
            x1, y1, x2, y2 = map(int, box)  # Chuyển đổi tọa độ box thành kiểu int
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Vẽ bounding box màu xanh
            cv2.putText(frame, person_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # Thêm tên

        # Ghi khung hình đã xử lý vào video đầu ra
        out.write(frame)

    # Giải phóng tài nguyên
    cap.release()
    out.release()
    print("Processing complete, output saved to:", output_path)

video_path = '/kaggle/input/test-data-sontung/sontung.mp4'
output_path = '/kaggle/working/output_video.mp4'
infer_video(resnet, video_path, output_path)
