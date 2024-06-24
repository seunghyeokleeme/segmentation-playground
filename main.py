from data_processing import extract_zip

def main():
    # 데이터 압축 해제
    zip_file_path = 'path_to_your_zip_file.zip'
    extract_to_path = 'path_to_extract'
    extract_zip(zip_file_path, extract_to_path)

if __name__ == "__main__":
    main()