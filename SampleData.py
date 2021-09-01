import copy
import os
import numpy as np
import cv2
import Utilis


class SampleData:
    KEY_NAME_LANDMARKS = "key_name_landMarks"
    KEY_NAME_X_MIN = "key_name_x_min"
    KEY_NAME_Y_MIN = "key_name_y_min"
    KEY_NAME_X_MAX = "key_name_x_max"
    KEY_NAME_Y_MAX = "key_name_y_max"
    NOR_MAT = np.array([[0.0, 0.0],
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 1.0]], np.float32)

    def __init__(self, image_name, full_image_path, detected_face_rect, land_mark_truth_obj):
        self._image_name = image_name
        self._full_image_path = full_image_path
        self._detected_face_rect = detected_face_rect
        self._land_mark_truth_obj = land_mark_truth_obj
        self._ferm_node_index = 0                # get A, B pixel val via feature cur pos
        self._predic_image = None
        self._train_image = None

        # decide normalize/unnormalize matrix, landmarks truth normalized matrix
        detected_face_left = detected_face_rect[0]
        detected_face_top = detected_face_rect[1]
        detected_face_width = detected_face_rect[2]
        detected_face_height = detected_face_rect[3]
        detected_face_coord_matrix = np.array([[detected_face_left, detected_face_top],
                                               [detected_face_left + detected_face_width, detected_face_top],
                                               [detected_face_left, detected_face_top + detected_face_height],
                                               [detected_face_left + detected_face_width, detected_face_top + detected_face_height]], np.float32)
        self._nor_matrix = Utilis.computeSimilarityTransform(detected_face_coord_matrix, SampleData.NOR_MAT)
        self._unnor_matrix = Utilis.computeSimilarityTransform(SampleData.NOR_MAT, detected_face_coord_matrix)
        self._mean_to_cur_normalized = None
        if self._land_mark_truth_obj is not None:
            self._landmark_truth_normalize = Utilis.translateTo(self.getLandmarkTruth(), self._nor_matrix)
        else:
            self._landmark_truth_normalize = None
        self._cur_landmark = None
        self._cur_landmark_normalize = None
        # print('verify, land_marks={}'.format(land_marks))
        # print('self._landmark_truth_normalize={}'.format(self._landmark_truth_normalize))
        # land_marks_back = SampleData.translateTo(self._landmark_truth_normalize, self._unnor_matrix)
        # print('verify, land_marks_back={}'.format(land_marks_back))

    @staticmethod
    def isFaceIn(face, x_min, y_min, x_max, y_max):
        (dec_face_x, dec_face_y, dec_face_w, dec_face_h) = face
        width = x_max - x_min
        height = y_max - y_min

        if dec_face_h > 1.5 * height or dec_face_h < 0.5 * height:
            return False
        elif dec_face_w > 1.5 * width or dec_face_w < 0.5 * width:
            return False
        lamda = 0.4
        if (dec_face_x < x_min - lamda * width) or (dec_face_x > x_min + lamda * width):
            return False
        elif (dec_face_y < y_min - lamda * height) or (dec_face_y > y_min + lamda * height):
            return False
        elif dec_face_x + dec_face_w < x_max - lamda * width or dec_face_x + dec_face_w > x_max + lamda * width:
            return False
        elif dec_face_y + dec_face_h < y_max - lamda * height or dec_face_y + dec_face_h > y_max + lamda * height:
            return False
        else:
            return True

    ###############################################################
    # return the landMark object like:
    # {
    # 'key_name_landMarks': array([[ 67.577102, 165.827455],
    #    						               [ 68.133929, 187.330357],
    #						                   ..
    #						                  ]),
    # 'key_name_x_min': 101.93605,
    # 'key_name_y_min': 133.073971,
    # 'key_name_x_max': 207.326226,
    # 'key_name_y_max': 231.229747
    # }
    ###############################################################
    @staticmethod
    def landMarkFrom(label_path):
        with open(label_path) as f:
            lines = f.readlines()
            # print('lines={}'.format(lines))
            num_landmarks = int(lines[1].split(" ")[2])
            landMarkObj = {}
            landMark = np.zeros((num_landmarks, 2))
            x_min = x_max = 0
            y_min = y_max = 0
            for idx in range(num_landmarks):
                x_y_string = lines[3 + idx].split(" ")
                x = float(x_y_string[0])
                y = float(x_y_string[1])
                landMark[idx, 0] = x
                landMark[idx, 1] = y
                if idx == 0:
                    x_min = x_max = x
                    y_min = y_max = y
                else:
                    x_min = min(x, x_min)
                    x_max = max(x, x_max)
                    y_min = min(y, y_min)
                    y_max = max(y, y_max)
                # print('x={}, y={}'.format(x, y))
            landMarkObj[SampleData.KEY_NAME_LANDMARKS] = landMark
            landMarkObj[SampleData.KEY_NAME_X_MIN] = x_min
            landMarkObj[SampleData.KEY_NAME_Y_MIN] = y_min
            landMarkObj[SampleData.KEY_NAME_X_MAX] = x_max
            landMarkObj[SampleData.KEY_NAME_Y_MAX] = y_max
            return landMarkObj
        return None

    ###############################################################
    # return the numpy array of SampleData objects [sampleData1, sampleData2, ..,]
    ###############################################################
    @staticmethod
    def samplesFrom(image_name, image_path, label_path):
        full_image_path = os.path.join(image_path, image_name + ".png")
        full_label_path = os.path.join(label_path, image_name + ".pts")
        image = cv2.imread(full_image_path)
        if image is None:
            return []
        land_mark_truth_obj = SampleData.landMarkFrom(full_label_path)
        if land_mark_truth_obj is None:
            return []

        faces_rect = Utilis.getFacesInRect(image,
                                    land_mark_truth_obj[SampleData.KEY_NAME_X_MIN],
                                    land_mark_truth_obj[SampleData.KEY_NAME_Y_MIN],
                                    land_mark_truth_obj[SampleData.KEY_NAME_X_MAX],
                                    land_mark_truth_obj[SampleData.KEY_NAME_Y_MAX])
        image = None
        if faces_rect is None:
            return []

        return [SampleData(image_name, full_image_path, face, land_mark_truth_obj) for face in faces_rect]


    def getLandmarkTruth(self):
        if self._land_mark_truth_obj is not None:
            return self._land_mark_truth_obj[SampleData.KEY_NAME_LANDMARKS]
        return None

    def getNomalizedLandmarkTruth(self):
        return self._landmark_truth_normalize

    def setNomalizedCurLandmark(self, src_normalized_landmark):
        self._cur_landmark_normalize = copy.deepcopy(src_normalized_landmark)
        self._cur_landmark = Utilis.translateTo(self._cur_landmark_normalize, self._unnor_matrix)

    def alignCurrentLandmark(self):
        image = cv2.imread(self._full_image_path)
        if image is None:
            return
        rows = image.shape[0]
        cols = image.shape[1]

        for idx_row in range(len(self._cur_landmark)):
            # print('self._cur_landmark({})={}'.format(idx_row, self._cur_landmark[idx_row][0]))
            if self._cur_landmark[idx_row, 0] < 0:
                self._cur_landmark[idx_row, 0] = 0
            elif self._cur_landmark[idx_row, 0] >= cols:
                self._cur_landmark[idx_row, 0] = cols - 1

            if self._cur_landmark[idx_row, 1] < 0:
                self._cur_landmark[idx_row, 1] = 0
            elif self._cur_landmark[idx_row, 1] >= rows:
                self._cur_landmark[idx_row, 1] = rows - 1

    def drawFaceRect(self, image):
        l = self._detected_face_rect[0]
        t = self._detected_face_rect[1]
        w = self._detected_face_rect[2]
        h = self._detected_face_rect[3]
        cv2.rectangle(image, (l, t), (l + w, t + h), (255, 255, 255))

    def _cvShowImg(self, image_name, image):
        w = 800
        h = 600
        cv2.namedWindow(image_name)
        cv2.resizeWindow(image_name, w, h)
        cv2.imshow(image_name, image)
        cv2.waitKey(500)

    ###############################################################
    # get image by read self._full_image_path first,
    # the use self._predic_image if the previous one is
    # None
    ###############################################################
    def _getImage(self):
        image = cv2.imread(self._full_image_path)
        if image is None:
            image = copy.deepcopy(self._predic_image)
        return image

    def show(self):
        # face rect
        image = self._getImage()
        if image is None:
            return

        # face rect
        self.drawFaceRect(image)

        # land mark truth
        landmark_truth = self.getLandmarkTruth()
        if landmark_truth is not None:
            numLandMarks = landmark_truth.shape[0]
            for iLandMark in range(numLandMarks):
                point = landmark_truth[iLandMark]
                cv2.circle(image, (int(point[0]), int(point[1])), 1, (0, 255, 255), 1)

        # land mark current
        if self._cur_landmark is not None:
            numLandMarks = self._cur_landmark.shape[0]
            for iLandMark in range(numLandMarks):
                point = self._cur_landmark[iLandMark]
                cv2.circle(image, (int(point[0]), int(point[1])), 1, (0, 122, 255), 2)

        self._cvShowImg(self._image_name, image)

    def showCurLandmarks(self):
        image = self._getImage()
        if image is None:
            return

        # face rect
        self.drawFaceRect(image)

        ll = 36
        lr = 41
        rl = 42
        rr = 47

        # current landmarks
        for iLandMark in range(self._cur_landmark.shape[0]):
            point = self._cur_landmark[iLandMark]
            if iLandMark >= ll and iLandMark <= lr:
                cv2.circle(image, (int(point[0]), int(point[1])), 1, (34, 0, 0), 2)
            elif iLandMark >= rl and iLandMark <= rr:
                cv2.circle(image, (int(point[0]), int(point[1])), 1, (134, 0, 0), 2)
            else:
                cv2.circle(image, (int(point[0]), int(point[1])), 1, (204, 0, 0), 2)

        self._cvShowImg(self._image_name, image)

    def showCurFaceAndFeature(self, feature_pool, feature_closest_landmark_no):
        image = self._getImage()
        if image is None:
            return

        want_feature_A_idx = 2
        want_feature_B_idx = 399

        # draw feature A (1, 2)
        want_feature_A_nor = np.expand_dims(feature_pool[want_feature_A_idx], 0)
        want_feature_A = Utilis.translateTo(want_feature_A_nor, self._unnor_matrix)
        want_feature_A_no = feature_closest_landmark_no[want_feature_A_idx]
        cv2.circle(image, (int(want_feature_A[0, 0]), int(want_feature_A[0, 1])), 1, (204, 180, 0), 4)

        # draw feature B (1, 2)
        want_feature_B_nor = np.expand_dims(feature_pool[want_feature_B_idx], 0)
        want_feature_B = Utilis.translateTo(want_feature_B_nor, self._unnor_matrix)
        want_feature_B_no = feature_closest_landmark_no[want_feature_B_idx]
        cv2.circle(image, (int(want_feature_B[0, 0]), int(want_feature_B[0, 1])), 1, (204, 200, 0), 4)

        # face rect
        self.drawFaceRect(image)

        # current landmarks
        for iLandMark in range(self._cur_landmark.shape[0]):
            point = self._cur_landmark[iLandMark]
            if want_feature_A_no == iLandMark:
                cv2.circle(image, (int(point[0]), int(point[1])), 1, (204, 180, 0), 2)
            elif want_feature_B_no == iLandMark:
                cv2.circle(image, (int(point[0]), int(point[1])), 1, (204, 200, 0), 2)
            else:
                cv2.circle(image, (int(point[0]), int(point[1])), 1, (204, 0, 0), 2)
        cv2.imshow('', image)

    def showMeanFaceAndFeature(self, mean_normalized, feature_pool, feature_closest_landmark_no):
        image = self._getImage()
        if image is None:
            return

        want_feature_A_idx = 2
        want_feature_B_idx = 399

        # draw feature A (1, 2)
        want_feature_A_nor = np.expand_dims(feature_pool[want_feature_A_idx], 0)
        want_feature_A = self.translateTo(want_feature_A_nor, self._unnor_matrix)
        want_feature_A_no = feature_closest_landmark_no[want_feature_A_idx]
        cv2.circle(image, (int(want_feature_A[0, 0]), int(want_feature_A[0, 1])), 1, (204, 180, 0), 4)

        # draw feature B (1, 2)
        want_feature_B_nor = np.expand_dims(feature_pool[want_feature_B_idx], 0)
        want_feature_B = self.translateTo(want_feature_B_nor, self._unnor_matrix)
        want_feature_B_no = feature_closest_landmark_no[want_feature_B_idx]
        cv2.circle(image, (int(want_feature_B[0, 0]), int(want_feature_B[0, 1])), 1, (204, 200, 0), 4)

        # face rect
        self.drawFaceRect(image)

        # mean landmarks
        mean_landmarks = self.translateTo(mean_normalized, self._unnor_matrix)
        for iLandMark in range(mean_landmarks.shape[0]):
            point = mean_landmarks[iLandMark]
            if want_feature_A_no == iLandMark:
                cv2.circle(image, (int(point[0]), int(point[1])), 1, (204, 180, 0), 2)
            elif want_feature_B_no == iLandMark:
                cv2.circle(image, (int(point[0]), int(point[1])), 1, (204, 200, 0), 2)
            else:
                cv2.circle(image, (int(point[0]), int(point[1])), 1, (204, 0, 0), 2)

        cv2_imshow(image)


def samplesFrom(img_name, images_path, labels_path):
    return SampleData.samplesFrom(img_name, images_path, labels_path)


def isFaceIn(face, x_min, y_min, x_max, y_max):
    return SampleData.isFaceIn(face, x_min, y_min, x_max, y_max)