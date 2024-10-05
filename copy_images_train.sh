#!/bin/bash

# 기본 경로 설정
SOURCE_DIR="DAD"                # 원본 DAD 폴더 경로
TARGET_DIR="DAD_C/training/frames"  # 타겟 폴더 경로

# 타겟 폴더가 없으면 생성
mkdir -p "$TARGET_DIR"

# 모든 Tester 폴더를 순회
for tester in "$SOURCE_DIR"/Tester*; do
    # Tester1, Tester2 등 폴더 이름을 추출
    tester_name=$(basename "$tester")

    # 각 Tester 폴더 안의 normal_driving 폴더를 순회
    for driving_folder in "$tester"/normal_driving_*; do
        # normal_driving_1, normal_driving_2 등 폴더 이름을 추출
        driving_name=$(basename "$driving_folder")
        
        # full path of front_IR folder
        front_ir_folder="$driving_folder/front_IR"
        
        # 타겟 폴더명 설정: 예) Tester1_normal_driving_1
        target_folder_name="${tester_name}_${driving_name}"
        target_folder="$TARGET_DIR/$target_folder_name"
        
        # 타겟 폴더가 없으면 생성
        mkdir -p "$target_folder"
        
        # front_IR 폴더의 모든 이미지 파일을 타겟 폴더로 복사
        if [ -d "$front_ir_folder" ]; then
            cp "$front_ir_folder"/*.png "$target_folder"/
            echo "Copied files from $front_ir_folder to $target_folder"
        else
            echo "Skipping $front_ir_folder: No such folder"
        fi
    done
done
