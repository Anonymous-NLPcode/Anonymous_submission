import os

def rename_files(folder_path):
    jpg_files = [file for file in os.listdir(folder_path) if file.lower().endswith('.jpg')]
    total_files = len(jpg_files)

    if total_files == 0:
        print("No .png files found in the folder.")
        return

    for idx, file in enumerate(jpg_files):
        old_path = os.path.join(folder_path, file)
        new_name = f"{idx + 1:03d}.png"
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)

    print(f"{total_files} files renamed successfully.")

# Replace 'folder_path' with the path to your folder containing .png files
folder_path = "my_food/005.my_food"
rename_files(folder_path)
