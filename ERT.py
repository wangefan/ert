import copy
import json
import Utilis
import numpy as np

from Configuration import Configuration
from Regressor import Regressor
from SampleData import SampleData


class ERT:
    def __init__(self,
                 configuration):
        self._configuration = configuration
        self._train_data = None
        self._validation_data = None
        self._mean_landmarks_normalized = None

        self._regressors = []
        for idx_regressor in range(self._configuration._cascade_number):
            regressor = Regressor(idx_regressor, self._configuration)
            self._regressors.append(regressor)

    def processData(self, train_data, validation_data):
        if train_data is None or validation_data is None or train_data[0].getLandmarkTruth() is None:
            return

        # calculate mean face
        rows_landmarks = train_data[0].getLandmarkTruth().shape[0]
        cols_landmarks = train_data[0].getLandmarkTruth().shape[1]
        self._mean_landmarks_normalized = np.zeros((rows_landmarks, cols_landmarks))
        for sampldata in train_data:
            # print('sampldata.getNomalizedLandmarkTruth()[0, 0] = {}'.format(sampldata.getNomalizedLandmarkTruth()[0, 0]))
            self._mean_landmarks_normalized += sampldata.getNomalizedLandmarkTruth()
            # print('self._mean_landmarks_normalized[0, 0] = {}'.format(self._mean_landmarks_normalized[0, 0]))
        self._mean_landmarks_normalized /= len(train_data)
        # print('self._mean_landmarks_normalized = {}'.format(self._mean_landmarks_normalized))

        self._train_data = Utilis.generateTrainDatas(train_data, self._configuration._train_data_times)

        Utilis.generateValidationDatas(validation_data, self._mean_landmarks_normalized)
        self._validation_data = validation_data

        if len(self._train_data) <= 0:
            print('ERT.processData() end, no training data')
        else:
            # self._train_data[1].show()
            # self._train_data[8].show()
            print('ERT.processData() end, num training data = {}, num Validation = {}'.format(len(self._train_data),
                                                                                              len(self._validation_data)))

    def train(self):
        print('ERT.train() begin')
        # self._train_data[0].show()
        for idx_regressor in range(len(self._regressors)):
            regressor = self._regressors[idx_regressor]
            regressor.train(self._train_data, self._validation_data, self._mean_landmarks_normalized)

            print('regressor {} train finished, compute error'.format(idx_regressor))
            error_train = Utilis.computeError(self._train_data)
            print('error_train = {}'.format(error_train))
            error_valid = Utilis.computeError(self._validation_data)
            print('error_valid = {}'.format(error_valid))

            self._validation_data[2].showCurLandmarks()
        print('ERT.train() end')

    MODEL_NAME = 'model_name'
    LANDMARK_NUM = 'landmarks_num'
    CASCADE_NUM = 'cascade_num'
    FERM_NUM = 'ferm_num'
    FERM_NUM_PER_GROUP = 'ferm_num_per_group'
    FERM_DEPTH = 'ferm_depth'
    FEATURE_POOL_NUM = 'feature_pool_num'
    CANDIDATE_FERM_NODE_INFO_NUM = 'candidate_ferm_node_infos_num'
    SHRINKAGE_FAC = 'shrinkage_factor'
    PADDING = 'padding'
    LAMDA = 'lamda'
    MEAN_FACE = 'mean_face'
    REGRESSORS = 'regressors'

    ###############################################################
    # return landmarks_cur(68, 2) coordinate by image.
    ###############################################################
    def predict(self, image):
        print('ERT.predict() egin')
        faces = Utilis.getFaces(image)
        the_1st_face = faces[0]
        fake_sample_data = SampleData('fake_img_name', 'fake_img_path', the_1st_face, None)
        fake_sample_data.setNomalizedCurLandmark(copy.deepcopy(self._mean_landmarks_normalized))
        fake_sample_data._predic_image = image

        for idx_regressor in range(len(self._regressors)):
            print('regressor {} predict begin'.format(idx_regressor))
            fake_sample_data.show()
            regressor = self._regressors[idx_regressor]
            regressor.predict(fake_sample_data, self._mean_landmarks_normalized)

            print('regressor {} predict finished'.format(idx_regressor))
        print('ERT.predict() end')

    ###############################################################
    # return landmarks_cur(68, 2) coordinate by image.
    ###############################################################
    def predictByEachRegressors(self, image):
        print('ERT.predict() egin')
        faces = Utilis.getFaces(image)
        the_1st_face = faces[0]
        fake_sample_data = SampleData('fake_img_name', 'fake_img_path', the_1st_face, None)
        fake_sample_data.setNomalizedCurLandmark(copy.deepcopy(self._mean_landmarks_normalized))
        fake_sample_data._predic_image = image

        for idx_regressor in range(len(self._regressors)):
            print('regressor {} predict begin'.format(idx_regressor))
            fake_sample_data.show()
            regressor = self._regressors[idx_regressor]
            regressor.predict(fake_sample_data, self._mean_landmarks_normalized)

            print('regressor {} predict finished'.format(idx_regressor))
        print('ERT.predict() end')


    def save(self, model_full_path):
        ert = {}
        # general info
        ert[ERT.MODEL_NAME] = 'yf_face_align_model'
        ert[ERT.CASCADE_NUM] = self._configuration._cascade_number
        ert[ERT.FERM_NUM] = self._configuration._ferm_number
        ert[ERT.FERM_NUM_PER_GROUP] = self._configuration._ferm_num_per_group
        ert[ERT.FERM_DEPTH] = self._configuration._ferm_depth
        ert[ERT.LANDMARK_NUM] = self._configuration._num_landmarks
        ert[ERT.FEATURE_POOL_NUM] = self._configuration._feature_pool_size
        ert[ERT.CANDIDATE_FERM_NODE_INFO_NUM] = self._configuration._num_candidate_ferm_node_infos
        ert[ERT.SHRINKAGE_FAC] = self._configuration._shrinkage_factor
        ert[ERT.PADDING] = self._configuration._padding
        ert[ERT.LAMDA] = self._configuration._lamda

        # mean face
        ert[ERT.MEAN_FACE] = self._mean_landmarks_normalized.tolist()

        # regressors
        ert[ERT.REGRESSORS] = []
        for idx_regressor in range(self._configuration._cascade_number):
            regressor = self._regressors[idx_regressor]
            if regressor is not None:
                ert[ERT.REGRESSORS].append(regressor.getJSONObj())

        with open(model_full_path, 'w') as outfile:
            json.dump(ert, outfile)

    @staticmethod
    def load(model_full_path):
        ert_load = None
        with open(model_full_path) as json_file:
            ert_load = json.load(json_file)
            # general info
            if ert_load is not None:
                # ert[ERT.MODEL_NAME] = 'yf_face_align_model'
                configuration = Configuration(ert_load[ERT.LANDMARK_NUM],
                                              1,
                                              ert_load[ERT.CASCADE_NUM],
                                              ert_load[ERT.FERM_NUM],
                                              ert_load[ERT.FERM_NUM_PER_GROUP],
                                              ert_load[ERT.FERM_DEPTH],
                                              ert_load[ERT.CANDIDATE_FERM_NODE_INFO_NUM],
                                              ert_load[ERT.FEATURE_POOL_NUM],
                                              ert_load[ERT.SHRINKAGE_FAC],
                                              ert_load[ERT.PADDING],
                                              ert_load[ERT.LAMDA])
                ert = ERT(configuration)

                # mean face
                ert._mean_landmarks_normalized = np.asarray(ert_load[ERT.MEAN_FACE])

                # regressors
                cascade_num = len(ert_load[ERT.REGRESSORS])
                for idx_regressor in range(cascade_num):
                    regressor = ert._regressors[idx_regressor]
                    regressor_json = ert_load[ERT.REGRESSORS][idx_regressor]
                    if regressor is not None and regressor_json is not None:
                        regressor.fromJSONOBJ(regressor_json)
                return ert

        return None
