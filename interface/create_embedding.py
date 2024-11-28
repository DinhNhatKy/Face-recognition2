import os
import tkinter
from tkinter import ttk, filedialog
import customtkinter
from infer.get_embedding import create_data_embeddings 
from CustomTkinterMessagebox import CTkMessagebox



folder_path = None
def create_embedding(root):
    new_window = customtkinter.CTkToplevel(root)
    new_window.geometry("500x600")
    new_window.title("Data Folder Explorer")

    frame_1 = customtkinter.CTkFrame(master=new_window)
    frame_1.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

    new_window.grid_rowconfigure(0, weight=1)
    new_window.grid_columnconfigure(0, weight=1)

    label = customtkinter.CTkLabel(master=frame_1, text="Folder Explorer")
    label.grid(row=0, column=0, pady=10, padx=10, sticky="n")

    bg_color = root._apply_appearance_mode(customtkinter.ThemeManager.theme["CTkFrame"]["fg_color"])
    text_color = root._apply_appearance_mode(customtkinter.ThemeManager.theme["CTkLabel"]["text_color"])
    selected_color = root._apply_appearance_mode(customtkinter.ThemeManager.theme["CTkButton"]["fg_color"])

    treestyle = ttk.Style()
    treestyle.theme_use('default')
    treestyle.configure("Treeview", background=bg_color, foreground=text_color, fieldbackground=bg_color, borderwidth=0)
    treestyle.map('Treeview', background=[('selected', bg_color)], foreground=[('selected', selected_color)])

    treeview = ttk.Treeview(frame_1, height=15, show="tree")
    treeview.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

    def populate_tree(parent, path):
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            item_id = treeview.insert(parent, 'end', text=item, open=False)
            if os.path.isdir(item_path):
                populate_tree(item_id, item_path)

    def load_folder():
        global folder_path
        folder_path = filedialog.askdirectory(title="Select dataset")
        if folder_path:
            embedding_button.configure(state="normal")
            for item in treeview.get_children():
                treeview.delete(item)
            populate_tree("", folder_path)
            return folder_path 

    def process_embedding():

        if folder_path:
            save_path = filedialog.askdirectory(title="Select save folder")
            if save_path:
                recognition_model_name = "inceptionresnetV1" 
                create_data_embeddings(folder_path, recognition_model_name, save_path)
        else: 
            CTkMessagebox.messagebox(title='Data folder', text='Please select datafolder!', sound='on', button_text='OK')

    load_button = customtkinter.CTkButton(master=frame_1, text="Upload data", command=load_folder)
    load_button.grid(row=2, column=0, pady=10, padx=10, sticky="n")

    embedding_button = customtkinter.CTkButton(master=frame_1, text="Create Embedding", command=process_embedding)
    embedding_button.grid(row=3, column=0, pady=10, padx=10, sticky="n")
    embedding_button.configure(state="disabled")

    frame_1.grid_rowconfigure(1, weight=1) 
    frame_1.grid_columnconfigure(0, weight=1)