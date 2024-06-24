import os
from PIL import Image, ImageOps
import numpy as np
import zipfile

def extract_zip(zip_file_path, extract_to_path):
    """압축풀기"""
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_path)
    print(f"{extract_to_path} 경로에 압축을 해제합니다.")

def process_image(label_image_path):
    """단일 사진 클래스 개수 확인."""
    # 이미지 로드 및 nupmy 배열로 변환
    label_image = Image.open(label_image_path)
    label_image_array = np.array(label_image)

    # 레이블 이미지의 고유값과 분포 시각화
    unique, counts = np.unique(label_image_array, return_counts=True)

    # 고유값 분포를 프린트
    print("Unique pixel values in label image:", unique)
    print("Counts for each value:", counts)
    return label_image, label_image_array

def get_image_paths(input_dir, target_dir) -> tuple[list, list]:
    input_img_paths = sorted([ os.path.join(input_dir, fname)
                        for fname in os.listdir(input_dir)
                        if fname.endswith('pre_disaster.png')])

    target_img_paths = sorted([ os.path.join(target_dir, fname)
                            for fname in os.listdir(target_dir)
                            if fname.endswith('pre_disaster_target.png')])
    
    return input_img_paths, target_img_paths
