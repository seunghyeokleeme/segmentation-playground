import os
import numpy as np
import random
from data_processing import extract_zip, process_image, get_image_paths, show
from dataset import xBD
from model import get_model
from k_fold import create_kfold_sets

def main():
    input_dir = 'train/images'
    target_dir = 'train/targets'
    img_size = (1024, 1024)
    num_classes = 1  # sigmoid 는 1로 설정해야 한다
    batch_size = 8

    # 데이터 압축 해제
    zip_file_path = 'path_to_your_zip_file.zip'
    extract_to_path = 'path_to_extract'
    extract_zip(zip_file_path, extract_to_path)

     # 이미지 경로 설정
    input_img_paths, target_img_paths = get_image_paths(input_dir, target_dir)

    label_image, label_image_array = process_image(target_img_paths[7])

    show(input_img_paths[7], target_img_paths[7])

    random.Random(1337).shuffle(input_img_paths)
    random.Random(1337).shuffle(target_img_paths)

    print(len(input_img_paths))
    print(len(target_img_paths))

    # KFold 설정
    num_folds = 28 # 예시 5
    epochs_per_fold = 20  # 각 fold마다 수행할 에폭 수

    # 배열을 num_folds분할
    input_splits = np.array_split(input_img_paths, num_folds)
    target_splits = np.array_split(target_img_paths, num_folds)

    # 검증용과 훈련용 데이터셋 생성
    validation_sets, train_input_sets, train_target_sets = create_kfold_sets(num_folds, input_splits, target_splits)

    # 결과 출력
    for i in range(num_folds):
        print(f"Fold {i+1}:")
        print(f"Validation Set - Input: {len(validation_sets[i][0])}, Target: {len(validation_sets[i][1])}")
        print(f"Train Set - Input: {len(train_input_sets[i])}, Target: {len(train_target_sets[i])}")
        print(f"First Validation Input: {validation_sets[i][0][0]}")
        print(f"First Validation Target: {validation_sets[i][1][0]}")
        print()

    # 체크포인트 및 백업 기록 폴더 생성
    result_dir = "path_to_your_result"
    checkpoint_dir = f"{result_dir}/model" #.h5파일 담을 폴더 설정
    history_dir = f"{result_dir}/histories" # history 파일 담을 폴더 설정
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(history_dir, exist_ok=True)
    

if __name__ == "__main__":
    main()