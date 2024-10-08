import os
import shutil
from pathlib import Path

def create_directory(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def restructure_training_data(src_root, dest_root):
    activities = {
        '001': 'normal_driving_1', '002': 'normal_driving_2', '003': 'normal_driving_3',
        '004': 'normal_driving_4', '005': 'normal_driving_5', '006': 'normal_driving_6',
    }
    
    for tester in range(1, 26):
        tester_id = f"{tester:02d}"
        for activity_code, activity_name in activities.items():
            for sensor in ['front_depth', 'front_IR', 'top_depth', 'top_IR']:
                src_path = os.path.join(src_root, f"Tester{tester}", activity_name, sensor)
                dest_path = os.path.join(dest_root, 'training', sensor, 'frames', f"{tester_id}_{activity_code}")
                
                if os.path.exists(src_path):
                    create_directory(dest_path)
                    for img in os.listdir(src_path):
                        shutil.copy2(os.path.join(src_path, img), os.path.join(dest_path, img))

def restructure_testing_data(src_root, dest_root):
    for val in range(1, 7):
        val_id = f"{val:02d}"
        for rec in range(1, 7):
            rec_id = f"{rec:04d}"
            for sensor in ['front_depth', 'front_IR', 'top_depth', 'top_IR']:
                src_path = os.path.join(src_root, f"val{val:02d}", f"rec{rec}", sensor)
                dest_path = os.path.join(dest_root, 'testing', sensor, 'frames', f"{val_id}_{rec_id}")
                
                if os.path.exists(src_path):
                    create_directory(dest_path)
                    for img in os.listdir(src_path):
                        shutil.copy2(os.path.join(src_path, img), os.path.join(dest_path, img))

def main():
    src_root = "DAD"  # 원본 데이터셋의 루트 디렉토리
    dest_root = "DAD_Jigsaw"  # 새로운 구조로 복사될 디렉토리
    
    for mode in ['training', 'testing']:
        for sensor in ['front_depth', 'front_IR', 'top_depth', 'top_IR']:
            create_directory(os.path.join(dest_root, mode, sensor, 'frames'))
    
    restructure_training_data(src_root, dest_root)
    restructure_testing_data(src_root, dest_root)

if __name__ == "__main__":
    main()