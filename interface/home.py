import tkinter as tk
from tkinter import Label
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
import time
from interface.load_image import load_and_display_images
from interface.create_embedding import create_embedding
from tkinter import filedialog
import os
from CustomTkinterMessagebox import CTkMessagebox
import yaml


with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)['infer_video']

antispoof_model = Fasnet()
cap = None

recogn_model = get_model('inceptionresnetV1')

embeddings = None
image2class= None 
index2class = None

def select_db():
    global  embeddings, image2class, index2class
    db_path = filedialog.askdirectory(title="please select db")

    embeddings_path= 'data/data_source/db1/inceptionresnetV1_embeddings.npy'
    image2class_path = 'data/data_source/db1/inceptionresnetV1_image2class.pkl'
    index2class_path = 'data/data_source/db1/inceptionresnetV1_index2class.pkl'

    if db_path:
        for file_name in os.listdir(db_path):
            if file_name.endswith('.npy'):
                embeddings_path = os.path.join(db_path, file_name)
            elif file_name.endswith('_image2class.pkl'):
                image2class_path= os.path.join(db_path, file_name)
            else:
                index2class_path = os.path.join(db_path, file_name)

    embeddings, image2class, index2class = load_embeddings_and_names(
                                                                    embedding_file_path= embeddings_path,
                                                                    image2class_file_path= image2class_path,
                                                                    index2class_file_path= index2class_path)


def infer_camera( 
                
                min_face_area=config['min_face_area'],
                bbox_threshold= config['bbox_threshold'], 
                required_images=config['required_images'],
                validation_threshold=config['validation_threshold'],
                is_anti_spoof=config['is_anti_spoof'], 
                is_vote=config['is_vote'], 
                distance_mode=config['distance_mode'], 
                anti_spoof_threshold=config['anti_spoof_threshold']):
    
    global embeddings, image2class, index2class

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Không thể mở camera")
        return

    valid_images = []
    is_reals = []
    previous_message = 0
    last_sound_time = 0
    sound_delay = 2 

    def update_frame():
        nonlocal previous_message, valid_images, is_reals, last_sound_time
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't capture frame")
                break

            input_image, face, prob, landmark = get_align(frame) 

            if face is not None:
                x1, y1, x2, y2 = map(int, face)
                if prob > bbox_threshold:
                    line_length = 20
                    color = (0, 255, 0) 
                    thickness = 1     

                    cv2.line(frame, (x1, y1), (x1 + line_length, y1), color, thickness) 
                    cv2.line(frame, (x1, y1), (x1, y1 + line_length), color, thickness) 

                    cv2.line(frame, (x2, y1), (x2 - line_length, y1), color, thickness) 
                    cv2.line(frame, (x2, y1), (x2, y1 + line_length), color, thickness) 

                    cv2.line(frame, (x1, y2), (x1 + line_length, y2), color, thickness) 
                    cv2.line(frame, (x1, y2), (x1, y2 - line_length), color, thickness)

                    cv2.line(frame, (x2, y2), (x2 - line_length, y2), color, thickness) 
                    cv2.line(frame, (x2, y2), (x2, y2 - line_length), color, thickness) 

                    cv2.putText(frame, f"Face {prob:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

                area = (face[2] - face[0]) * (face[3] - face[1])

                if prob > bbox_threshold:
                    center = np.mean(landmark, axis=0)
                    height, width, _ = frame.shape
                    center_x, center_y = center

                    distance_from_center = np.sqrt((center_x - width / 2) ** 2 + (center_y - height / 2) ** 2)
                    current_time = time.time()

                    if area > min_face_area:
                        if width * 0.15 < center_x < width * 0.85 and height * 0.15 < center_y < height * 0.85 and distance_from_center < min(width, height) * 0.4:
                            if previous_message != 1 and current_time - last_sound_time > sound_delay:
                                threading.Thread(target=playsound, args=('audio/guide_keepface.mp3',), daemon=True).start()
                                last_sound_time = current_time
                                previous_message = 1

                            is_real, score = antispoof_model.analyze(frame, map(int, face)) 
                            print(is_real, score)
                            is_reals.append((is_real, score))
                            valid_images.append(input_image)

                        else:
                            if previous_message != 2 and current_time - last_sound_time > sound_delay:
                                threading.Thread(target=playsound, args=('audio/guide_centerface.mp3',), daemon=True).start()
                                last_sound_time = current_time
                                previous_message = 2

                    else:
                        if previous_message != 3 and current_time - last_sound_time > sound_delay:
                            threading.Thread(target=playsound, args=('audio/closer.mp3',), daemon=True).start()
                            last_sound_time = current_time
                            previous_message = 3

            else:
                if previous_message != 0:
                    print("No face")
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
                print(f"Collected enough {required_images} valid images")
                result = {
                    'valid_images': valid_images,
                    'is_reals': is_reals
                }
                canvas.delete('all')
                person_name = check_validation(
                    result, 
                    embeddings, 
                    image2class, 
                    index2class, 
                    recogn_model,
                    validation_threshold=validation_threshold,
                    is_anti_spoof=is_anti_spoof,
                    is_vote=is_vote,
                    distance_mode=distance_mode,
                    anti_spoof_threshold = anti_spoof_threshold
                )
                update_status_display(person_name) 
                break

        cap.release()

    threading.Thread(target=update_frame, daemon=True).start()


def start_infer_camera():
    if embeddings is None:
        CTkMessagebox.messagebox(title='Database select', text='Please select database!', sound='on', button_text='OK')
    else: 
        min_face_area_value = int(min_face_area_slider.get())
        bbox_threshold_value = float(bbox_threshold_slider.get())
        num_image_value = int(num_image_slider.get())
        validation_threshold_value = float(validation_threshold_slider.get())
        is_anti_spoof_value = anti_spoof_switch.get()
        is_vote_value = vote_switch.get()
        distance_mode_value = distance_mode_optionmenu.get()
        anti_spoof_threshold = float(anti_spoof_threshold_slider.get())
        infer_camera(
                        min_face_area=min_face_area_value,
                        bbox_threshold=bbox_threshold_value,
                        required_images=num_image_value,
                        validation_threshold=validation_threshold_value,
                        is_anti_spoof=is_anti_spoof_value,
                        is_vote=is_vote_value,
                        distance_mode=distance_mode_value,
                        anti_spoof_threshold= anti_spoof_threshold
                    )

def update_status_display(person_name):
    if person_name:  
        image_path = 'audio/accept.png'
        message = f"Accepted: {person_name}"
    else: 
        image_path = 'audio/reject.png'
        message = "Rejected"

    original_image_label.configure(text=message)

    or_image = Image.open(image_path)
    or_image.thumbnail((250, 250))
    origin_image = ImageTk.PhotoImage(or_image)
    status_image_label.configure(image=origin_image)
    status_image_label.image = origin_image

    def reset_status():
        original_image_label.configure(text="Waiting for result...")
        status_image_label.configure(image=None)
        status_image_label.image = None

    original_image_label.after(4000, reset_status)





# START INTERFACE 
root = customtkinter.CTk()
root.title("FACE RECOGNITION")
root.geometry(f"900x550")
# root.resizable(False, False) 
root.grid_columnconfigure((1, 2, 3), weight=1)
root.grid_rowconfigure((0, 1, 2,), weight=1)



################### Menu ####################
menubar = tk.Menu(root)

file = tk.Menu(menubar, tearoff=0)
file.add_separator()
file.add_command(label='Exit', command=root.destroy)
menubar.add_cascade(label='Main', menu=file)

# Upload and arguementation
edit = tk.Menu(menubar, tearoff=0)
edit.add_command(label='open',  command= lambda:  load_and_display_images(root))
menubar.add_cascade(label='Upload image', menu=edit)

# Create embedding
tools_menu = tk.Menu(menubar, tearoff=0)
tools_menu.add_command(label="Create embedding", command= lambda:  create_embedding(root))
menubar.add_cascade(label="Embedding", menu=tools_menu)



################## Layout ###################
left_frame = customtkinter.CTkFrame(root, width=140, corner_radius=0)
left_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
left_frame.grid_rowconfigure(4, weight=1)


right_frame = customtkinter.CTkFrame(root, width=140, corner_radius=0)
right_frame.grid(row=1, column=4, rowspan=3, sticky="nsew", padx=(0, 10), pady=(0, 10))
right_frame.grid_rowconfigure(4, weight=1)

person_name_frame = customtkinter.CTkFrame(root, width=140, corner_radius=0)
person_name_frame.grid(row=0, column=4, rowspan=1, sticky="nsew", padx=(0, 10), pady=(10, 10))
person_name_frame.grid_rowconfigure(1, weight=1)


original_image_label = customtkinter.CTkLabel(person_name_frame, text="Waiting for detection...", anchor="center")
original_image_label.grid(row=0, column=0, sticky="nsew")

status_image_label = Label(person_name_frame)
status_image_label.grid(row=1, column=0, sticky="nsew")

person_name_frame.columnconfigure(0, weight=1)
person_name_frame.rowconfigure((0, 1), weight=1)



################ Left frame wiget ##########################
logo_label = customtkinter.CTkLabel(left_frame, text="TOOLBAR", font=customtkinter.CTkFont(size=20, weight="bold"))
logo_label.pack(pady=7)

button_open = customtkinter.CTkButton(left_frame, text="Open Camera", width=110, height=30, command=start_infer_camera)
button_open.pack(pady=7)


min_face_area_label = customtkinter.CTkLabel(left_frame, text="Min_face_area: 10000") 
min_face_area_label.pack(pady=7)

def update_minface_value(value):
    min_face_area_label.configure(text=f"Min_face_area: {int(value)}")

min_face_area_slider = customtkinter.CTkSlider(left_frame, from_=5000, to=50000, number_of_steps=1000,
                              command=update_minface_value)  
min_face_area_slider.set(10000)  
min_face_area_slider.pack(pady=5, padx=(4,4))


bbox_threshold_label = customtkinter.CTkLabel(left_frame, text="Bbox_threshold: 0.7") 
bbox_threshold_label.pack(pady=5)

def update_bbox_threhold(value):
    bbox_threshold_label.configure(text=f"Bbox_threshold: {value:.2f}")

bbox_threshold_slider = customtkinter.CTkSlider(left_frame, from_=0.5, to=1.0, number_of_steps=10, command= update_bbox_threhold)
bbox_threshold_slider.set(0.7)
bbox_threshold_slider.pack(pady=4, padx=(4,4))


anti_spoof_threshold_label = customtkinter.CTkLabel(left_frame, text="Anti spoof threshold: 0.7") 
anti_spoof_threshold_label.pack(pady=5)

def update_anti_spoof_thres(value):
    bbox_threshold_label.configure(text=f"Anti spoof threshold: {value:.2f}")

anti_spoof_threshold_slider = customtkinter.CTkSlider(left_frame, from_=0.5, to=1.0, number_of_steps=10, command= update_anti_spoof_thres)
anti_spoof_threshold_slider.set(0.9)
anti_spoof_threshold_slider.pack(pady=4, padx=(4,4))


num_image_label = customtkinter.CTkLabel(left_frame, text="Num_image: 16") 
num_image_label.pack(pady=4)

def update_num_image(value):
    num_image_label.configure(text=f"Num image: {int(value)}")

num_image_slider = customtkinter.CTkSlider(left_frame, from_=1, to=50, number_of_steps=50, command= update_num_image)
num_image_slider.set(16)
num_image_slider.pack(pady=4, padx=(4,4))


anti_spoof_switch_label = customtkinter.CTkLabel(left_frame, text="Is anti_spoof") 
anti_spoof_switch_label.pack(pady=4)

anti_spoof_switch = customtkinter.CTkSwitch(left_frame, text="Enable")
anti_spoof_switch.pack(pady=4)


is_vote_label = customtkinter.CTkLabel(left_frame, text="Is vote") 
is_vote_label.pack(pady=4)

vote_switch = customtkinter.CTkSwitch(left_frame, text="Enable")
vote_switch.pack(pady=4)


distance_mode_optionmenu = customtkinter.CTkOptionMenu(left_frame, values=["cosine", "l2"])
distance_mode_optionmenu.pack( padx=4, pady=(4, 4))

validation_threshold_label = customtkinter.CTkLabel(left_frame, text="validation_threshold: 0.7") 
validation_threshold_label.pack(pady=4)

def update_validation_threshold(value):
    validation_threshold_label.configure(text=f"validation_threshold: {value:.2f}")

validation_threshold_slider = customtkinter.CTkSlider(left_frame, from_=0.5, to=1.0, number_of_steps=20, command= update_validation_threshold)
validation_threshold_slider.set(0.7)
validation_threshold_slider.pack(pady=4, padx=(4,4))




################ Right frame wiget ##########################
button_select_db = customtkinter.CTkButton(right_frame, text="Select db", width=110, height=30, command=select_db)
button_select_db.grid(row=6, column=0, padx=20, pady=(10, 20))

def change_appearance_mode(new_appearance_mode: str):
    customtkinter.set_appearance_mode(new_appearance_mode)

appearance_mode_optionmenu = customtkinter.CTkOptionMenu(right_frame, values=["Light", "Dark", "System"], command=change_appearance_mode)
appearance_mode_optionmenu.grid(row=7, column=0, padx=20, pady=(10, 10))

scaling_label = customtkinter.CTkLabel(right_frame, text="UI Scaling:", anchor="w")
scaling_label.grid(row=8, column=0, padx=20, pady=(10, 0))

def change_scaling(new_scaling: str):
    new_scaling_float = int(new_scaling.replace("%", "")) / 100
    customtkinter.set_widget_scaling(new_scaling_float)

scaling_optionmenu = customtkinter.CTkOptionMenu(right_frame, values=["80%", "90%", "100%", "110%", "120%"], command= change_scaling)
scaling_optionmenu.grid(row=9, column=0, padx=20, pady=(10, 20))

canvas = customtkinter.CTkCanvas(root, background="gray", borderwidth=2, relief="solid")
canvas.grid(row=0, column=1, columnspan=3, rowspan=3, sticky="nsew", padx=30, pady=30)

root.config(menu=menubar)
root.mainloop()
