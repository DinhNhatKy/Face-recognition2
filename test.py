import os

def rename_subfolders(upload_folder_path):
    # Kiểm tra nếu thư mục upload_folder_path tồn tại
    if not os.path.exists(upload_folder_path):
        print(f"Thư mục {upload_folder_path} không tồn tại.")
        return
    
    # Lấy danh sách các thư mục con trong thư mục tải lên
    subfolders = [f.path for f in os.scandir(upload_folder_path) if f.is_dir()]
    
    for folder in subfolders:
        # Lấy tên thư mục
        folder_name = os.path.basename(folder)
        
        # Thay đổi tên thư mục (ví dụ chuyển thành chữ hoa)
        new_folder_name = folder_name.upper()
        
        # Tạo đường dẫn mới cho thư mục
        new_folder_path = os.path.join(upload_folder_path, new_folder_name)
        
        # Đổi tên thư mục
        os.rename(folder, new_folder_path)
        print(f"Đã thay đổi tên thư mục từ {folder_name} thành {new_folder_name}")

# Ví dụ sử dụng
upload_folder_path = 'C:\Users\KyDN\VN_CELEB\casi'  # Thay đổi đường dẫn này theo thực tế
rename_subfolders(upload_folder_path)
