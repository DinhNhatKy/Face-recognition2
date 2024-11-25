import tkinter as tk
import customtkinter
from PIL import Image, ImageTk
import cv2
import numpy as np
from playsound import playsound
from models.spoofing.FasNet import Fasnet
from infer.infer_image import get_align
from infer.infer_video import check_validation, load_embeddings_and_names
from infer.utils import get_model
import threading

# Khởi tạo các biến toàn cục
antispoof_model = Fasnet()
cap = None
previous_message = None
valid_images = []
is_reals = []
recogn_model = get_model('inceptionresnetV1')
embeddings, image2class, index2class = load_embeddings_and_names(
    'data/data_source/inceptionresnetV1_embeddings.npy',
    'data/data_source/inceptionresnetV1_image2class.pkl',
    'data/data_source/inceptionresnetV1_index2class.pkl')

is_infer_running = False

def infer_camera(min_face_area=10000, bbox_threshold=0.7, required_images=16):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Không thể mở camera")
        return

    valid_images = []
    is_reals = []
    previous_message = 0

    def update_frame():
        nonlocal previous_message, valid_images, is_reals
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Không thể chụp được hình ảnh")
                break

            input_image, face, prob, landmark = get_align(frame)
            if face is not None:
                x1, y1, x2, y2 = map(int, face)
                if prob > bbox_threshold:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Face {prob:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                area = (face[2] - face[0]) * (face[3] - face[1])

                if prob > bbox_threshold:
                    center = np.mean(landmark, axis=0)
                    height, width, _ = frame.shape
                    center_x, center_y = center

                    distance_from_center = np.sqrt((center_x - width / 2) ** 2 + (center_y - height / 2) ** 2)
                    if area > min_face_area:
                        if width * 0.15 < center_x < width * 0.85 and height * 0.15 < center_y < height * 0.85 and distance_from_center < min(width, height) * 0.4:
                            if previous_message != 1:
                                threading.Thread(target=playsound, args=('audio/guide_keepface.mp3',), daemon=True).start()
                                previous_message = 1
                            
                            is_real, score = antispoof_model.analyze(frame, map(int, face))
                            print(is_real, score)
                            is_reals.append((is_real, score))
                            valid_images.append(input_image)

                        else:
                            if previous_message != 2:
                                threading.Thread(target=playsound, args=('audio/guide_centerface.mp3',), daemon=True).start()
                                previous_message = 2

                    else:
                        if previous_message != 3:
                            print("Đưa khuôn mặt lại gần hơn")
                            previous_message = 3

            else:
                if previous_message != 0:
                    print("Không phát hiện khuôn mặt")
                    previous_message = 0


            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            image_tk = ImageTk.PhotoImage(image)

            canvas.delete("all")
            canvas.create_image(canvas_width // 2, canvas_height // 2, image=image_tk, anchor="center")
            canvas.image = image_tk

            if len(valid_images) >= required_images:
                print(f"Đã thu thập đủ {required_images} ảnh hợp lệ.")
                result = {
                    'valid_images': valid_images,
                    'is_reals': is_reals
                }
                check_validation(result, embeddings, image2class, index2class, recogn_model)
                break

        cap.release()

    # Khởi chạy vòng lặp trong luồng riêng
    threading.Thread(target=update_frame, daemon=True).start()





def change_appearance_mode(new_appearance_mode: str):
    customtkinter.set_appearance_mode(new_appearance_mode)


def change_scaling(new_scaling: str):
    new_scaling_float = int(new_scaling.replace("%", "")) / 100
    customtkinter.set_widget_scaling(new_scaling_float)



root = customtkinter.CTk()
root.title("FACE RECOGNITION")
root.geometry(f"900x550")
root.resizable(False, False) 
root.grid_columnconfigure((1, 2, 3), weight=1)
root.grid_rowconfigure((0, 1, 2,), weight=1)

left_frame = customtkinter.CTkFrame(root, width=140, corner_radius=0)
left_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
left_frame.grid_rowconfigure(4, weight=1)

right_frame = customtkinter.CTkFrame(root, width=140, corner_radius=0)
right_frame.grid(row=1, column=4, rowspan=3, sticky="nsew", padx=(0, 10), pady=(0, 10))
right_frame.grid_rowconfigure(4, weight=1)

logo_label = customtkinter.CTkLabel(left_frame, text="TOOLBAR", font=customtkinter.CTkFont(size=20, weight="bold"))
logo_label.pack(pady=7)

button_open = customtkinter.CTkButton(left_frame, text="Open Camera", width=110, height=30, command=infer_camera)
button_open.pack(pady=7)




min_face_area_label = customtkinter.CTkLabel(left_frame, text="Min_face_area: 10000")  # Giá trị mặc định
min_face_area_label.pack(pady=7)

def update_minface_value(value):
    min_face_area_label.configure(text=f"Min_face_area: {int(value)}")

min_face_area = customtkinter.CTkSlider(left_frame, from_=5000, to=50000, number_of_steps=1000,
                              command=update_minface_value)  
min_face_area.set(10000)  
min_face_area.pack(pady=5, padx=(4,4))


bbox_threshold_label = customtkinter.CTkLabel(left_frame, text="Bbox_threshold: 0.7")  # Giá trị mặc định
bbox_threshold_label.pack(pady=5)

def update_bbox_threhold(value):
    bbox_threshold_label.configure(text=f"Bbox_threshold: {value:.2f}")

bbox_threshold = customtkinter.CTkSlider(left_frame, from_=0.5, to=1.0, number_of_steps=10, command= update_bbox_threhold)
bbox_threshold.set(0.7)
bbox_threshold.pack(pady=4, padx=(4,4))



num_image_label = customtkinter.CTkLabel(left_frame, text="Num_image: 16")  # Giá trị mặc định
num_image_label.pack(pady=4)

def update_num_image(value):
    num_image_label.configure(text=f"Num image: {int(value)}")

num_image = customtkinter.CTkSlider(left_frame, from_=1, to=50, number_of_steps=50, command= update_num_image)
num_image.set(16)
num_image.pack(pady=4, padx=(4,4))


is_anti_spoof = False

def on_is_spoof_change(value):
    global is_anti_spoof 
    is_anti_spoof = value  


anti_spoof_switch_label = customtkinter.CTkLabel(left_frame, text="Is anti_spoof") 
anti_spoof_switch_label.pack(pady=4)

anti_spoof_switch = customtkinter.CTkSwitch(left_frame, text="Enable", command=lambda: on_is_spoof_change(anti_spoof_switch.get()))
anti_spoof_switch.pack(pady=4)


is_vote = False

def on_if_vote_change(value):
    global is_vote 
    is_vote = value  


is_vote_label = customtkinter.CTkLabel(left_frame, text="Is vote") 
is_vote_label.pack(pady=4)

vote_switch = customtkinter.CTkSwitch(left_frame, text="Enable", command=lambda: on_if_vote_change(vote_switch.get()))
vote_switch.pack(pady=4)


distance_mode = 'Cosine'

def change_distance_mode(value):
    global distance_mode  # Tham chiếu đến biến toàn cục
    distance_mode = value
    print(f"Distance mode changed to: {distance_mode}")

appearance_mode_optionmenu = customtkinter.CTkOptionMenu(left_frame, values=["l2", "cosine"], command=change_distance_mode)
appearance_mode_optionmenu.pack( padx=4, pady=(4, 4))



valid_threshold_label = customtkinter.CTkLabel(left_frame, text="Valid_threshold: 0.7")  # Giá trị mặc định
valid_threshold_label.pack(pady=4)

def update_valid_threshold(value):
    valid_threshold_label.configure(text=f"Valid_threshold: {value:.2f}")

valid_threhold = customtkinter.CTkSlider(left_frame, from_=0.5, to=1.0, number_of_steps=20, command= update_valid_threshold)
valid_threhold.set(0.7)
valid_threhold.pack(pady=4, padx=(4,4))

# is_antispoof= False, validation_threhold= 0.7, is_vote= False, distance_mode = 'cosine'




appearance_mode_optionmenu = customtkinter.CTkOptionMenu(right_frame, values=["Light", "Dark", "System"], command=change_appearance_mode)
appearance_mode_optionmenu.grid(row=7, column=0, padx=20, pady=(10, 10))

scaling_label = customtkinter.CTkLabel(right_frame, text="UI Scaling:", anchor="w")
scaling_label.grid(row=8, column=0, padx=20, pady=(10, 0))

scaling_optionmenu = customtkinter.CTkOptionMenu(right_frame, values=["80%", "90%", "100%", "110%", "120%"], command= change_scaling)
scaling_optionmenu.grid(row=9, column=0, padx=20, pady=(10, 20))



canvas = customtkinter.CTkCanvas(root, background="gray", borderwidth=2, relief="solid")
canvas.grid(row=0, column=1, columnspan=3, rowspan=3, sticky="nsew", padx=30, pady=30)

root.mainloop()
