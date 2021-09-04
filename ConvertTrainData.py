import Utilis
import os
import SampleData
import cv2

def _cvShowImg(self, image_name, image):
    w = 800
    h = 600
    cv2.namedWindow(image_name)
    cv2.resizeWindow(image_name, w, h)
    cv2.imshow(image_name, image)
    cv2.waitKey(1000)

root = r"./lfpw"
train_path = root + r"/trainset_small"

train_images_path = train_path + r"/images"
train_labels_path = train_path + r"/labels"

file_names = Utilis.get_files_name(train_images_path)
for file_name in file_names:
    ori_label_path = os.path.join(train_labels_path, file_name + ".pts")
    new_file_name = file_name + '_ver2'
    new_label_path = os.path.join(train_labels_path, new_file_name + ".pts")
    with open(ori_label_path) as f_in, open(new_label_path, 'w') as f_out:
        picked_indicies = [4, 5, 6, 7, 8, 9, 10, 11, 12, 36, 39, 42, 45]
        lines = f_in.readlines()
        f_out.write(lines[0])
        f_out.write('n_points:  {}\r\n'.format(len(picked_indicies)))
        f_out.write('{\r\n')
        num_landmarks = int(lines[1].split(" ")[2])
        landmarks_list = []

        for idx in range(num_landmarks):
            if idx in picked_indicies:
                x_y_string = lines[3 + idx]
                f_out.write(x_y_string)
        f_out.write('}')
dest_file_name = 'image_0001'
dest_image_path = os.path.join(train_images_path, dest_file_name + ".png")
dest_label_path = os.path.join(train_labels_path, dest_file_name + '_ver2' + ".pts")
landmarks = SampleData.lanmarksFrom(dest_label_path)['key_name_landMarks']
image = cv2.imread(dest_image_path)
Utilis.drawLandmarks(landmarks, image, 3, (255, 55, 255))
w = 800
h = 600

cv2.imshow('1', image)
cv2.waitKey()