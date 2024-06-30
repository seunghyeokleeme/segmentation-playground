from dataset import xBD

def model_evaluate(model, batch_size, img_size, input_img_paths, target_img_paths):
    test_gen = xBD(batch_size, img_size, input_img_paths, target_img_paths)
    return model.evaluate(test_gen)
