from tkinter import filedialog
from PIL import Image, ImageTk
import os
import random
from torchvision import transforms
import customtkinter as ctk
from tkinter import messagebox

folder_name = None
max_columns= None

def load_and_display_images(root):
    list_images = []
    transformed_images = []
   
   
    tab_frame = ctk.CTkFrame(root)
    tab_frame.grid(row=0, column=1, columnspan=3, rowspan=3, sticky="nsew", padx=30, pady=30)
    tab_frame.grid_rowconfigure(0, weight=1)
    tab_frame.grid_columnconfigure(0, weight=1)

    custom_frame = ctk.CTkFrame(root, width=140, corner_radius=0)
    custom_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
    custom_frame.grid_rowconfigure(4, weight=1)

    canvas = ctk.CTkCanvas(tab_frame)
    canvas.grid(row=0, column=0, sticky="nsew")


    scrollbar = ctk.CTkScrollbar(tab_frame, orientation="vertical", command=canvas.yview)
    scrollbar.grid(row=0, column=1, sticky="ns")
    canvas.configure(yscrollcommand=scrollbar.set)


    image_frame = ctk.CTkFrame(canvas)
    canvas.create_window((0, 0), window=image_frame, anchor="nw")

    def update_scroll_region(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    image_frame.bind("<Configure>", update_scroll_region)


    def update_size(event):
        global max_columns
        width = canvas.winfo_width()
        max_columns = width // 100

    canvas.bind("<Configure>", update_size)
    

    def browse_folder():
        
        folder_path = filedialog.askdirectory(title="Chọn thư mục chứa ảnh")
        if not folder_path:
            return
        
        global folder_name 
        folder_name = folder_path.split('/')[-1]

        for widget in image_frame.winfo_children():
            widget.destroy()

        list_images.clear()
        transformed_images.clear()
        row, col = 0, 0

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
                if col >= max_columns-1:
                    col = 0
                    row += 1
        transform_button.configure(state="normal")

    def transform_images(list_images, k):
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor()
        ])
        transformed_images = []
            
        for image in list_images:
            image =  transforms.ToTensor()(image)
            transformed_images.append(image)

        for i in range(k- len(list_images)):
            img = random.choice(list_images)
            transformed_img = transform(img)
            transformed_images.append(transformed_img)

        return transformed_images

    def save_images(transformed_images):
        
        save_folder = filedialog.askdirectory(title="Chọn thư mục để lưu ảnh")
        if not save_folder:
            return
        

        new_folder_path = os.path.join(save_folder, folder_name)
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)


        for idx, transformed_img in enumerate(transformed_images):
            save_path = os.path.join(save_folder, f"{folder_name}/transformed_image_{idx+1}.png")
            transformed_img_pil = transforms.ToPILImage()(transformed_img)
            transformed_img_pil.save(save_path)
        messagebox.showinfo("Successful", f"Save {len(transformed_images)} image to {new_folder_path}")


    def on_transform():
        num_transform = int(num_transform_image.get())
        nonlocal transformed_images
        transformed_images = transform_images(list_images, num_transform)

        for widget in image_frame.winfo_children():
            widget.destroy()

        row, col = 0, 0

        print('len transform:', len(transformed_images))

        for img_tensor in transformed_images:
            img_pil = transforms.ToPILImage()(img_tensor)
            img_tk = ctk.CTkImage(img_pil, size=(100, 100))

            img_label = ctk.CTkLabel(image_frame, image=img_tk, text="")
            img_label.image = img_tk
            img_label.grid(row=row, column=col, padx=5, pady=5)

            col += 1
            if col >= max_columns-1:
                col = 0
                row += 1

        save_button.configure(state="normal")

    def close_window():
       
        image_frame.grid_forget()
        custom_frame.grid_forget()
        tab_frame.grid_forget()


    browse_button = ctk.CTkButton(custom_frame, text="Select folder", command=browse_folder)
    browse_button.grid(row=0, column=0, pady=(40, 10), padx=(20, 20))

    transform_button = ctk.CTkButton(custom_frame, text="Transform images", command=on_transform)
    transform_button.grid(row=1, column=0, pady=10, padx=(20, 20))
    transform_button.configure(state="disabled")
    
    save_button = ctk.CTkButton(custom_frame, text="Save images", command=lambda: save_images(transformed_images))
    save_button.grid(row=2, column=0, pady=10, padx=(20, 20))
    save_button.configure(state="disabled")

    close_button = ctk.CTkButton(custom_frame, text="Close", command=close_window)
    close_button.grid(row=3, column=0, pady=10, padx=(20, 20))



    transform_image_frame = ctk.CTkFrame(custom_frame)
    transform_image_frame.grid(row=4, column=0, pady=10, padx=(5, 5)) 


    label = ctk.CTkLabel(transform_image_frame, text="Select Number of Transformations:")
    label.pack(pady= 5)

    num_transform_image = ctk.CTkOptionMenu(transform_image_frame, values=['10', '20', '30', '40', '50', '60', '70', '80'])
    num_transform_image.pack(pady= 5)
