import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import random
from torchvision import transforms

def load_and_display_images(custom_frame, tab_frame):
    list_images = []
    transformed_images = []

    for widget in custom_frame.winfo_children():
        widget.destroy()

    image_frame = ctk.CTkFrame(tab_frame)
    image_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

    def browse_folder():
        folder_path = filedialog.askdirectory(title="Chọn thư mục chứa ảnh")
        if not folder_path:
            return

        for widget in image_frame.winfo_children():
            widget.destroy()

        list_images.clear()
        transformed_images.clear()
        row, col = 0, 0
        max_columns = 6

        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                file_path = os.path.join(folder_path, filename)

                img = Image.open(file_path)
                list_images.append(img)
                img.thumbnail((100, 100))
                img_tk = ctk.CTkImage(img, size=(100, 100))

                img_label = ctk.CTkLabel(image_frame, image=img_tk, text="")
                img_label.image = img_tk
                img_label.grid(row=row, column=col, padx=5, pady=5)

                col += 1
                if col >= max_columns:
                    col = 0
                    row += 1

    def transform_images(list_images, k):
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor()
        ])

        transformed_images = []

        for i in range(k):
            img = random.choice(list_images)
            transformed_img = transform(img)
            transformed_images.append(transformed_img)

        return transformed_images

    def save_images(transformed_images):
        save_folder = filedialog.askdirectory(title="Chọn thư mục để lưu ảnh")
        if not save_folder:
            return

        for idx, transformed_img in enumerate(transformed_images):
            save_path = os.path.join(save_folder, f"transformed_image_{idx+1}.png")
            transformed_img_pil = transforms.ToPILImage()(transformed_img)
            transformed_img_pil.save(save_path)

    def on_transform():
        num_transform = int(num_transform_image.get())
        nonlocal transformed_images
        transformed_images = transform_images(list_images, num_transform)

        for widget in image_frame.winfo_children():
            widget.destroy()

        row, col = 0, 0
        max_columns = 6
        for img in list_images:
            img.thumbnail((100, 100))
            img_tk = ctk.CTkImage(img, size=(100, 100))

            img_label = ctk.CTkLabel(image_frame, image=img_tk, text="")
            img_label.image = img_tk
            img_label.grid(row=row, column=col, padx=5, pady=5)

            col += 1
            if col >= max_columns:
                col = 0
                row += 1

        for img_tensor in transformed_images:
            img_pil = transforms.ToPILImage()(img_tensor)
            img_tk = ctk.CTkImage(img_pil, size=(100, 100))

            img_label = ctk.CTkLabel(image_frame, image=img_tk, text="")
            img_label.image = img_tk
            img_label.grid(row=row, column=col, padx=5, pady=5)

            col += 1
            if col >= max_columns:
                col = 0
                row += 1

        save_button.configure(state="normal")

    def close_window():
        for widget in custom_frame.winfo_children():
            widget.destroy()
        for widget in tab_frame.winfo_children():
            widget.destroy()

    # Sử dụng grid thay vì pack cho các button và option menu
    browse_button = ctk.CTkButton(custom_frame, text="Select folder", command=browse_folder)
    browse_button.grid(row=0, column=0, pady=(40, 10), padx=(20, 20))

    transform_button = ctk.CTkButton(custom_frame, text="Transform images", command=on_transform)
    transform_button.grid(row=1, column=0, pady=10, padx=(20, 20))

    save_button = ctk.CTkButton(custom_frame, text="Save images", command=lambda: save_images(transformed_images))
    save_button.grid(row=2, column=0, pady=10, padx=(20, 20))
    save_button.configure(state="disabled")

    close_button = ctk.CTkButton(custom_frame, text="Close", command=close_window)
    close_button.grid(row=3, column=0, pady=10, padx=(20, 20))

    num_transform_image = ctk.CTkOptionMenu(custom_frame, values=['10', '20', '30', '40', '50', '60', '70', '80'])
    num_transform_image.grid(row=4, column=0, pady=10, padx=(20, 20))
