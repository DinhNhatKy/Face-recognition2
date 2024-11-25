import cv2
import customtkinter as ctk
from tkinter import *
from PIL import Image, ImageTk
import threading

# Hàm mở camera và hiển thị video
def open_camera():
    cap = cv2.VideoCapture(0)  # Mở camera

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Chuyển đổi khung hình từ BGR sang RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Chuyển frame từ mảng numpy thành ảnh có thể hiển thị
        frame_image = Image.fromarray(frame_rgb)
        frame_photo = ImageTk.PhotoImage(frame_image)

        # Hiển thị hình ảnh lên canvas
        canvas.create_image(0, 0, image=frame_photo, anchor="nw")
        canvas.image = frame_photo  # Lưu giữ hình ảnh tham chiếu

        # Cập nhật giao diện
        window.update_idletasks()
        window.update()

    cap.release()

# Khởi tạo cửa sổ ứng dụng
window = ctk.CTk()

# Cài đặt giao diện
window.title("Ứng dụng Camera")
window.geometry("640x480")

# Tạo canvas để hiển thị video
canvas = ctk.CTkCanvas(window, width=640, height=480)
canvas.pack()

# Nút để mở camera
start_button = ctk.CTkButton(window, text="Bắt đầu camera", command=lambda: threading.Thread(target=open_camera, daemon=True).start())
start_button.pack()

# Chạy giao diện
window.mainloop()
