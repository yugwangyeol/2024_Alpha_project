import os

def rename_images_in_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith("img_") and file.endswith(".png"):
                file_path = os.path.join(root, file)
                file_number = file.split('_')[1].split('.')[0] 
                
                new_file_name = f"{int(file_number):04d}.png"
                new_file_path = os.path.join(root, new_file_name)
                
                os.rename(file_path, new_file_path)
                print(f"Renamed: {file_path} -> {new_file_path}")

def process_all_folders(root_dir):

    sensors = ['front_IR', 'front_depth', 'top_depth', 'top_IR']
    for mode in ['training', 'testing']:
        for sensor in sensors:
            frames_directory = os.path.join(root_dir, mode, sensor, 'frames')
            if os.path.exists(frames_directory):
                print(f"Processing: {frames_directory}")
                rename_images_in_directory(frames_directory)

root_dir = "DAD_Jigsaw" 

process_all_folders(root_dir)
