import numpy as np
import random

import Utilis
from Ferm import Ferm


class Regressor:
    def __init__(self,
                 no,
                 configuration):
        self._no = no
        self._configuration = configuration

        self._feature_pool = np.zeros((self._configuration._feature_pool_size, 2))
        self._feature_closest_landmark_offset = np.zeros((self._configuration._feature_pool_size, 2))
        self._feature_closest_landmark_no = np.zeros(self._configuration._feature_pool_size, dtype=int)

        self._ferms = []
        for idx_ferm in range(self._configuration._ferm_number):
            ferm = Ferm(idx_ferm, self._configuration)
            self._ferms.append(ferm)

    def computeSimilarityTransformFromMeanToCur(self, mean, datas):
        for sampleData in datas:
            sampleData._mean_to_cur_normalized = Utilis.computeSimilarityTransform(mean,
                                                                                       sampleData._cur_landmark_normalize)

    def generateFeaturePool(self, mean_landmarks_normalized):
        x_min, x_max, y_min, y_max = Utilis.getLandmarkMaxMin(mean_landmarks_normalized)
        padding = self._configuration._padding
        x_min -= padding
        x_max += padding
        y_min -= padding
        y_max += padding

        # print('self._feature_pool[399] = {}'.format(self._feature_pool[399]))
        # print('self._feature_closest_landmark_no[399] = {}'.format(self._feature_closest_landmark_no[399]))
        for idx_feature in range(self._configuration._feature_pool_size):
            self._feature_pool[idx_feature, 0] = random.uniform(x_min, x_max)
            self._feature_pool[idx_feature, 1] = random.uniform(y_min, y_max)
            closest_landmark_no = Utilis.getClosestLandmarkNo(mean_landmarks_normalized, self._feature_pool[idx_feature])
            self._feature_closest_landmark_offset[idx_feature] = self._feature_pool[idx_feature] - \
                                                                 mean_landmarks_normalized[closest_landmark_no]
            self._feature_closest_landmark_no[idx_feature] = closest_landmark_no

    def train(self, train_data, validation_data, mean_landmarks_normalized):
        print('>Regressor {} begin to train()..'.format(self._no))
        # print('Regressor.train(), train_data[0]._mean_to_cur_normalized = {}'.format(train_data[0]._mean_to_cur_normalized))
        # print('Regressor.train(), validation_data[0]._mean_to_cur_normalized = {}'.format(validation_data[0]._mean_to_cur_normalized))
        self.computeSimilarityTransformFromMeanToCur(mean_landmarks_normalized, train_data)
        self.computeSimilarityTransformFromMeanToCur(mean_landmarks_normalized, validation_data)
        # print('Regressor.train(), train_data[0]._mean_to_cur_normalized = {}'.format(train_data[0]._mean_to_cur_normalized))
        # print('Regressor.train(), validation_data[0]._mean_to_cur_normalized = {}'.format(validation_data[0]._mean_to_cur_normalized))

        self.generateFeaturePool(mean_landmarks_normalized)

        train_datas_leaf_info_in_group = []
        valid_datas_leaf_info_in_group = []
        for ferm in self._ferms:
            ferm.train(train_data,
                       validation_data,
                       self._feature_pool,
                       self._feature_closest_landmark_offset,
                       self._feature_closest_landmark_no)
            train_datas_leafs_no = Utilis.getDataLeafsNo(ferm, train_data)
            valid_datas_leafs_no = Utilis.getDataLeafsNo(ferm, validation_data)
            train_data_ferm_leaf_info_dic = {
                "FERM": ferm,
                "datas_leafs_no": train_datas_leafs_no
            }
            valid_data_ferm_leaf_info_dic = {
                "FERM": ferm,
                "datas_leafs_no": valid_datas_leafs_no
            }
            train_datas_leaf_info_in_group.append(train_data_ferm_leaf_info_dic)
            valid_datas_leaf_info_in_group.append(valid_data_ferm_leaf_info_dic)
            Utilis.resetDataLeafsIndex(train_data)
            Utilis.resetDataLeafsIndex(validation_data)
            if len(train_datas_leaf_info_in_group) >= self._configuration._ferm_num_per_group:
                Utilis.adjustCurLandmarks(train_data, train_datas_leaf_info_in_group, self._configuration)
                Utilis.adjustCurLandmarks(validation_data, valid_datas_leaf_info_in_group, self._configuration)
                train_datas_leaf_info_in_group = []
                valid_datas_leaf_info_in_group = []

        validation_data[2].showCurFaceAndFeature(self._feature_pool, self._feature_closest_landmark_no)

    REGRESSOR_NAME = 'regressor_name'
    REGRESSOR_NAME_VAL = 'regressor'
    FERMS = 'ferms'

    def predict(self, data, mean_landmarks_normalized):
        # get TS from mean to cur normalized
        data._mean_to_cur_normalized = Utilis.computeSimilarityTransform(mean_landmarks_normalized,
                                                                         data._cur_landmark_normalize)
        # predict in ferms
        datas_leaf_info_in_group = []
        datas = [data]
        for ferm in self._ferms:
            ferm.predict(datas)
            datas_leafs_no = Utilis.getDataLeafsNo(ferm, datas)
            datas_ferm_leaf_info_dic = {
                "FERM": ferm,
                "datas_leafs_no": datas_leafs_no
            }

            datas_leaf_info_in_group.append(datas_ferm_leaf_info_dic)
            Utilis.resetDataLeafsIndex(datas)

            if len(datas_leaf_info_in_group) >= self._configuration._ferm_num_per_group:
                Utilis.adjustCurLandmarks(datas, datas_leaf_info_in_group, self._configuration)
                datas_leaf_info_in_group = []


    def getJSONObj(self):
        result = {}
        regressor_name_val = '{}_{}'.format(Regressor.REGRESSOR_NAME_VAL, self._no)
        result[Regressor.REGRESSOR_NAME] = regressor_name_val
        result[Regressor.FERMS] = []
        for ferm in self._ferms:
            result[Regressor.FERMS].append(ferm.getJSONObj())
        return result

    def fromJSONOBJ(self, json_obj):
        # print('regressor json_obj = {}'.format(json_obj))
        ferms_json = json_obj[Regressor.FERMS]
        for idx_ferm_json in range(len(ferms_json)):
            ferm_json = ferms_json[idx_ferm_json]
            ferm = self._ferms[idx_ferm_json]
            ferm.fromJSONOBJ(ferm_json)
