import os
import zipfile

def extract_zip(zip_file_path, extract_to_path):
    """압축풀기"""
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_path)
    print(f"{extract_to_path} 경로에 압축을 해제합니다.")
