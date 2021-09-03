import cv2

from ERT import ERT
import Utilis

root = r"./"
model_full_path = root + r"lfpw/ert_model.json"
test_path = root + r"lfpw/testset_/"
test_images_path = test_path + r"images/image_0001.png"

# 1.Load model
print('1. Load model')
ert = ERT.load(model_full_path)

print('2. predict by each regressors')
test_data2_image = cv2.imread(test_images_path, cv2.IMREAD_GRAYSCALE)
ert.predictByEachRegressors(test_data2_image)
