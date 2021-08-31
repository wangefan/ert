from os import listdir
from pathlib import Path
import copy
import random
import math
import cv2
import numpy as np

import SampleData

# get files in path and save name in dest_files_name
# as array
def get_files_name(files_path, dest_files_name):
    files = listdir(files_path)
    for f in files:
        dest_files_name.append(Path(f).stem)


###############################################################
# return the array of SampleData objects [sampleData1, sampleData2, ..,]
###############################################################
def load_data(images_path, labels_path, dest_datas):
    datas_name = []
    get_files_name(images_path, datas_name)
    for img_name in datas_name:
        sampleDatas = SampleData.samplesFrom(img_name, images_path, labels_path)
        if sampleDatas is not None:
            for data in sampleDatas:
                dest_datas.append(data)


###############################################################
# return the numpy array face rect[l, t, w, h] like:
# [ 80  98 151 151]
###############################################################
def getFaces(image):
    root = r"./"
    face_model_path = root + r"CascadeClassifier/haarcascade_frontalface_alt2.xml"
    face_model = cv2.CascadeClassifier(face_model_path)
    faces = face_model.detectMultiScale(image,
                                        1.1,
                                        2,
                                        0,
                                        (30, 30))
    return faces


###############################################################
# return the numpy array face rect[l, t, w, h] like:
# [ 80  98 151 151]
###############################################################
def getFacesInRect(image, x_min, y_min, x_max, y_max):
    faces = getFaces(image)
    return [face for face in faces if SampleData.isFaceIn(face, x_min, y_min, x_max, y_max)]


###############################################################
# return the array of SampleData objects [sampleData1, sampleData2, ..,]
###############################################################
def generateTrainDatas(train_data, train_data_times):
    size = len(train_data) * train_data_times
    result = [None] * size
    for idx_org_train in range(len(train_data)):
        for idx_times in range(train_data_times):
            result_sample = copy.deepcopy(train_data[idx_org_train])
            idx_random = random.randrange(idx_org_train + 1, idx_org_train + len(train_data)) % len(train_data)
            random_sample = train_data[idx_random]
            result_sample.setNomalizedCurLandmark(random_sample.getNomalizedLandmarkTruth())
            result_sample.alignCurrentLandmark()
            result[idx_times * len(train_data) + idx_org_train] = result_sample

    return result


def translateTo(src_matrix, ts_matrix):
    rows = src_matrix.shape[0]  # m
    src_matrix_append_one_col = np.concatenate([src_matrix, np.ones((rows, 1))], 1)  # shape(m, n+1)
    result = np.matmul(src_matrix_append_one_col, ts_matrix)
    return result

###############################################################
# return the result matrix TS, that:
# src_matrix x TS = dst_matrix
# ex:
# (68, 3) x (3, 2) = (68, 2)
###############################################################
def computeSimilarityTransform(src_matrix, dest_matrix):
    rows = src_matrix.shape[0]  # m
    src_matrix_append_one_col = np.concatenate([src_matrix, np.ones((rows, 1))], 1)  # shape(m, n+1)
    pinv = np.zeros((src_matrix_append_one_col.shape[1], src_matrix_append_one_col.shape[0]))  # pinv (n+1, m)
    cv2.invert(src_matrix_append_one_col, pinv, cv2.DECOMP_SVD)
    result = np.matmul(pinv, dest_matrix)  # (3, 2)
    return result

###############################################################
# set current landmark/normalized data in validaion_datas
###############################################################
def generateValidationDatas(validaion_datas, mean_landmarks_normalized):
    for sample_data in validaion_datas:
        sample_data.setNomalizedCurLandmark(mean_landmarks_normalized)
        sample_data.alignCurrentLandmark()


###############################################################
# given numpy landmarks(num_landmarks, 2)
# return (x_min, x_max, y_min, y_max)
###############################################################
def getLandmarkMaxMin(landmarks):
    if landmarks is None:
        return None
    rows = landmarks.shape[0]
    x_min = x_max = landmarks[0, 0]
    y_min = y_max = landmarks[0, 1]
    for idx_row in range(1, rows):
        x_min = min(landmarks[idx_row, 0], x_min)
        x_max = max(landmarks[idx_row, 0], x_max)
        y_min = min(landmarks[idx_row, 1], y_min)
        y_max = max(landmarks[idx_row, 1], y_max)

    return (x_min, x_max, y_min, y_max)


def getDistance(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


###############################################################
# given mean_landmarks_normalized(num_landmarks, 2)
# for each landmark compute dist with feature point,
# return the landmark index of minimum distance with
# feature point
###############################################################
def getClosestLandmarkNo(mean_landmarks_normalized, feature_point):
    # print('mean_landmarks_normalized.shape = {}'.format(mean_landmarks_normalized.shape))
    # print('feature_point.shape = {}'.format(feature_point.shape))
    # print('feature_point = {}'.format(feature_point))
    landmark_rows = mean_landmarks_normalized.shape[0]
    no_min = 0
    pt_landmark = mean_landmarks_normalized[no_min]
    # print('pt_landmark = {}'.format(pt_landmark))
    # dist_min = math.sqrt((pt_landmark[0] - feature_point[0])**2 + (pt_landmark[1] - feature_point[1])**2)
    dist_min = getDistance(pt_landmark, feature_point)
    # print('dist_min = {}'.format(dist_min))
    for idx_landmark_no in range(1, landmark_rows):
        pt_landmark = mean_landmarks_normalized[idx_landmark_no]
        # dist = math.sqrt((pt_landmark[0] - feature_point[0])**2 + (pt_landmark[1] - feature_point[1])**2)
        dist = getDistance(pt_landmark, feature_point)
        if dist < dist_min:
            dist_min = dist
            no_min = idx_landmark_no
    return int(no_min)

###############################################################
# input: data list, ex[sample data1, sample data2,..]
# output: data list, ex[leaf no of sample data1, leaf no of sample data2,..]
###############################################################
def getDataLeafsNo(ferm, datas):
    result = []
    for data in datas:
        # print('getDataLeafsNo, data._ferm_node_index={}, ferm.getNodesNum()={}'.format(data._ferm_node_index, ferm.getNodesNum()))
        result.append(data._ferm_node_index - ferm.getNodesNum())
    # print('getDataLeafsNo, result={}'.format(result))
    return result

###############################################################
# reset _ferm_node_index of given datas
###############################################################
def resetDataLeafsIndex(datas):
    for data in datas:
        data._ferm_node_index = 0

###############################################################
# adjust current landmark/landmark normalized
# by data leaf info in group
###############################################################
def adjustCurLandmarks(datas, datas_leaf_info_in_group, configuration):
    for idx_data in range(len(datas)):
        # print('==adjustCurLandmarks, idx_data = {}'.format(idx_data))
        data = datas[idx_data]
        data_residual = np.zeros((configuration._num_landmarks, 2))
        for datas_leaf_info in datas_leaf_info_in_group:
            # print('datas_leaf_info = {}'.format(datas_leaf_info))
            ferm = datas_leaf_info["FERM"]
            datas_leafs_no = datas_leaf_info["datas_leafs_no"][idx_data]
            # print('datas_leafs_no = {}'.format(datas_leafs_no))
            itr_residual = ferm.getResidual(datas_leafs_no)
            # print('itr_residual = {}'.format(getPartsMatrix(itr_residual)))
            data_residual += itr_residual
            # print('data_residual = {}'.format(getPartsMatrix(data_residual)))
        data_residual /= len(datas_leaf_info_in_group)
        # print('mean data_residual = {}'.format(getPartsMatrix(data_residual)))
        temp_cur_landmark_normalize = copy.deepcopy(data._cur_landmark_normalize)
        temp_cur_landmark_normalize += configuration._shrinkage_factor * data_residual
        # print('configuration._shrinkage_factor = {}'.format(configuration._shrinkage_factor))
        # print('temp_cur_landmark_normalize = {}'.format(getPartsMatrix(temp_cur_landmark_normalize)))
        # print('data._cur_landmark_normalize = {}'.format(getPartsMatrix(data._cur_landmark_normalize)))
        data.setNomalizedCurLandmark(temp_cur_landmark_normalize)
        # print('after setting, data._cur_landmark_normalize = {}'.format(getPartsMatrix(data._cur_landmark_normalize)))
        # print('**adjustCurLandmarks, idx_data = {} end'.format(idx_data))


###############################################################
# return parts of source matrix
# ex: src_matrix = (68, 2)
# return result = (2, 2)
###############################################################
def getPartsMatrix(src_matrix):
    if src_matrix is not None:
        return src_matrix[:2, :]
    return None


def getDotSelf(mat):
    return np.sum(np.multiply(mat, mat))


def printAllDataFermNodeIndecis(datas):
    for data in datas:
        print('data._ferm_node_index = {}'.format(data._ferm_node_index))


def computeError(datas):
    ll = 36
    lr = 41
    rl = 42
    rr = 47
    total_error = 0.
    for data in datas:
        l = np.zeros((2,))
        r = np.zeros((2,))
        for idx_row_landmark in range(ll, lr + 1):
            l += data._cur_landmark[idx_row_landmark]
        l /= (lr - ll + 1)
        for idx_row_landmark in range(rl, rr + 1):
            r += data._cur_landmark[idx_row_landmark]
        r /= (rl - rr + 1)

        dist = getDistance(l, r)
        data_error = 0.
        for idx_landmark_row in range(data._cur_landmark.shape[0]):
            landmark_dist = getDistance(data.getLandmarkTruth()[idx_landmark_row], data._cur_landmark[idx_landmark_row])
            data_error += landmark_dist
        data_error /= dist
        total_error += data_error
    return total_error / len(datas)