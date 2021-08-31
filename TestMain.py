import cv2

from ERT import ERT
import Utilis

root = r"./"
model_full_path = root + r"lfpw/ert_model_small.json"
test_path = root + r"lfpw/testset_small/"
test_images_path = test_path + r"images"
test_labels_path = test_path + r"labels"

# 1.Load model
print('1. Load model')
ert = ERT.load(model_full_path)

# 2.Load sample data
print('2. Load sample data')
test_data = []
Utilis.load_data(test_images_path, test_labels_path, test_data)
test_data[1].show()

print('3. test data[2]')
test_data2_image = cv2.imread(test_data[1]._full_image_path, cv2.IMREAD_GRAYSCALE)
ert.predict(test_data2_image)
