from data_processing import extract_zip, process_image, get_image_paths

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

    print(input_img_paths[7])
    print(target_img_paths[7])

    label_image, label_image_array = process_image(target_img_paths[7])
    

if __name__ == "__main__":
    main()