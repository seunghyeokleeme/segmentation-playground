from data_processing import extract_tar, get_image_paths, show_predict, show
from dataset import xBD, xBDInference
from evaluate import model_evaluate
from tensorflow.keras.models import load_model

# .h5 파일로부터 모델 로드
model = load_model('path_to_your_model_file.h5') # 적절하게 경로 변경하시면 됩니다.
model.summary()

input_dir = 'test/images'
target_dir = 'test/targets'
img_size = (1024, 1024)
batch_size = 8

# 데이터 압축 해제
tar_file_path = 'path_to_your_tar_file.tar'
extract_to_path = 'path_to_extract'
extract_tar(tar_file_path, extract_to_path)

# 이미지 경로 설정
input_img_paths, _ = get_image_paths(input_dir, target_dir)
# input_img_paths, target_img_paths = get_image_paths(input_dir, target_dir)

test_input_img_paths = input_img_paths
# test_target_img_paths = target_img_paths

# model_evaluate(model, batch_size, img_size, test_input_img_paths, test_target_img_paths)

test_gen = xBDInference(batch_size, img_size, test_input_img_paths)

test_preds = model.predict(test_gen)

# show(test_input_img_paths[12], test_target_img_paths[12])
show_predict(12, test_preds)