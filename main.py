from data_processing import extract_zip, process_image

def main():
    # 데이터 압축 해제
    zip_file_path = 'path_to_your_zip_file.zip'
    extract_to_path = 'path_to_extract'
    extract_zip(zip_file_path, extract_to_path)

    label_image, label_image_array = process_image("사진경로/train/targets/guatemala-volcano_00000000_pre_disaster_target.png")
    

if __name__ == "__main__":
    main()