import math
import random
import copy

import cv2
import numpy as np


###############################################################
# Ferm structure, if depth = 3:
# num_ferm_node = 2^(3-1)-1 = 3    (index = 0, 1, 2)
# num_ferm_leaf = 2^(3-1) = 4      (index = 3, 4, 5, 6)
#                       0         ferm node
#                     /   \
#    						     1     2      ferm node
#						        / \    / \
#						       3  4   5   6   ferm leaf
###############################################################
import Utilis


class Ferm:
    A_FEATURE_CLOSEST_LANDMARK_NO = "a_feature_closest_landmark_no"
    B_FEATURE_CLOSEST_LANDMARK_NO = "b_feature_closest_landmark_no"
    A_FEATURE_CLOSEST_LANDMARK_OFFSET_X = "a_feature_closest_landmark_offset_x"
    A_FEATURE_CLOSEST_LANDMARK_OFFSET_Y = "a_feature_closest_landmark_offset_y"
    B_FEATURE_CLOSEST_LANDMARK_OFFSET_X = "b_feature_closest_landmark_offset_x"
    B_FEATURE_CLOSEST_LANDMARK_OFFSET_Y = "b_feature_closest_landmark_offset_y"
    FEATURE_THRESHOLD = "feature_threshold"
    FERM_NODE_INFO = {
        A_FEATURE_CLOSEST_LANDMARK_NO: 0,
        B_FEATURE_CLOSEST_LANDMARK_NO: 0,
        A_FEATURE_CLOSEST_LANDMARK_OFFSET_X: 0,
        A_FEATURE_CLOSEST_LANDMARK_OFFSET_Y: 0,
        B_FEATURE_CLOSEST_LANDMARK_OFFSET_X: 0,
        B_FEATURE_CLOSEST_LANDMARK_OFFSET_Y: 0,
        FEATURE_THRESHOLD: 0
    }

    def __init__(self,
                 no,
                 configuration):
        self._no = no
        self._configuration = configuration
        num_ferm_node = int(math.pow(2, self._configuration._ferm_depth - 1) - 1)
        num_ferm_leaf = int(math.pow(2, self._configuration._ferm_depth - 1))

        self._ferm_nodes = [copy.deepcopy(Ferm.FERM_NODE_INFO) for idx in range(num_ferm_node)]

        # single ferm leaf incluse residual (num_landmarks, 2), ex: ferm_leafs[0].shape = (68, 2)
        self._ferm_leafs = [np.zeros((self._configuration._num_landmarks, 2)) for idx in range(num_ferm_leaf)]

    def getNodesNum(self):
        if self._ferm_nodes is not None:
            return len(self._ferm_nodes)
        return 0

    def getResidual(self, leaf_no):
        if self._ferm_leafs is not None:
            return self._ferm_leafs[leaf_no]
        return None

    def splitNodeForPredict(self,
                  idx_ferm_node,
                  datas,
                  ferm_node_info):
        for data in datas:
            if data._ferm_node_index == idx_ferm_node:
                # get A, B pixel val via feature cur pos
                if data._predic_image is None:
                    continue

                # get A, B feautre pos in cur landmark
                A_feature_closest_landmark_offset_mean_face = np.array([(ferm_node_info[
                                                                             Ferm.A_FEATURE_CLOSEST_LANDMARK_OFFSET_X],
                                                                         ferm_node_info[
                                                                             Ferm.A_FEATURE_CLOSEST_LANDMARK_OFFSET_Y])])
                B_feature_closest_landmark_offset_mean_face = np.array([(ferm_node_info[
                                                                             Ferm.B_FEATURE_CLOSEST_LANDMARK_OFFSET_X],
                                                                         ferm_node_info[
                                                                             Ferm.B_FEATURE_CLOSEST_LANDMARK_OFFSET_Y])])
                A_feature_closest_landmark_offset_cur_nor = np.matmul(A_feature_closest_landmark_offset_mean_face,
                                                                      data._mean_to_cur_normalized[:2, :])
                B_feature_closest_landmark_offset_cur_nor = np.matmul(B_feature_closest_landmark_offset_mean_face,
                                                                      data._mean_to_cur_normalized[:2, :])
                A_feature_closest_landmark_no = ferm_node_info[Ferm.A_FEATURE_CLOSEST_LANDMARK_NO]
                B_feature_closest_landmark_no = ferm_node_info[Ferm.B_FEATURE_CLOSEST_LANDMARK_NO]
                A_feature_cur_nor = A_feature_closest_landmark_offset_cur_nor + data._cur_landmark_normalize[
                    A_feature_closest_landmark_no]
                B_feature_cur_nor = B_feature_closest_landmark_offset_cur_nor + data._cur_landmark_normalize[
                    B_feature_closest_landmark_no]
                A_feature_cur = Utilis.translateTo(A_feature_cur_nor, data._unnor_matrix).astype(int)
                B_feature_cur = Utilis.translateTo(B_feature_cur_nor, data._unnor_matrix).astype(int)



                rows = data._predic_image.shape[0]
                cols = data._predic_image.shape[1]
                A_feature_pixel = B_feature_pixel = 0
                if A_feature_cur[0, 0] >= 0 and A_feature_cur[0, 0] < cols and A_feature_cur[0, 1] >= 0 and \
                        A_feature_cur[0, 1] < rows:
                    A_feature_pixel = float(data._predic_image[A_feature_cur[0, 1], A_feature_cur[0, 0]])
                if B_feature_cur[0, 0] >= 0 and B_feature_cur[0, 0] < cols and B_feature_cur[0, 1] >= 0 and \
                        B_feature_cur[0, 1] < rows:
                    B_feature_pixel = float(data._predic_image[B_feature_cur[0, 1], B_feature_cur[0, 0]])

                threshold = ferm_node_info[Ferm.FEATURE_THRESHOLD]

                # get pixel diff and split it
                left_node_index = 2 * idx_ferm_node + 1
                right_node_index = 2 * idx_ferm_node + 2
                if (A_feature_pixel - B_feature_pixel) > threshold:
                    data._ferm_node_index = left_node_index
                else:
                    data._ferm_node_index = right_node_index

    def splitNode(self,
                  idx_ferm_node,
                  datas,
                  ferm_node_info,
                  get_score=False):
        landmarks_num = self._configuration._num_landmarks
        mean_left = np.zeros((landmarks_num, 2))
        mean_right = np.zeros((landmarks_num, 2))
        num_left = num_right = 0
        # if get_score:
        #  print('splitNode get score, iterate all data begin..')
        # else:
        #  print('splitNode!, iterate all data begin..')
        for data in datas:
            if data._ferm_node_index == idx_ferm_node:
                # get A, B feautre pos in cur landmark
                A_feature_closest_landmark_offset_mean_face = np.array([(ferm_node_info[
                                                                             Ferm.A_FEATURE_CLOSEST_LANDMARK_OFFSET_X],
                                                                         ferm_node_info[
                                                                             Ferm.A_FEATURE_CLOSEST_LANDMARK_OFFSET_Y])])
                B_feature_closest_landmark_offset_mean_face = np.array([(ferm_node_info[
                                                                             Ferm.B_FEATURE_CLOSEST_LANDMARK_OFFSET_X],
                                                                         ferm_node_info[
                                                                             Ferm.B_FEATURE_CLOSEST_LANDMARK_OFFSET_Y])])
                # print('B_feature_closest_landmark_offset_mean_face.shape = {}'.format(B_feature_closest_landmark_offset_mean_face.shape))
                # print('B_feature_closest_landmark_offset_mean_face = {}'.format(B_feature_closest_landmark_offset_mean_face))
                # print('data._mean_to_cur_normalized[:2, :] = {}'.format(data._mean_to_cur_normalized[:2, :]))
                A_feature_closest_landmark_offset_cur_nor = np.matmul(A_feature_closest_landmark_offset_mean_face,
                                                                      data._mean_to_cur_normalized[:2, :])
                B_feature_closest_landmark_offset_cur_nor = np.matmul(B_feature_closest_landmark_offset_mean_face,
                                                                      data._mean_to_cur_normalized[:2, :])
                # print('B_feature_closest_landmark_offset_cur_nor.shape = {}'.format(B_feature_closest_landmark_offset_cur_nor.shape))
                # print('B_feature_closest_landmark_offset_cur_nor = {}'.format(B_feature_closest_landmark_offset_cur_nor))
                A_feature_closest_landmark_no = ferm_node_info[Ferm.A_FEATURE_CLOSEST_LANDMARK_NO]
                B_feature_closest_landmark_no = ferm_node_info[Ferm.B_FEATURE_CLOSEST_LANDMARK_NO]
                # print('B_feature_closest_landmark_no = {}'.format(B_feature_closest_landmark_no))
                A_feature_cur_nor = A_feature_closest_landmark_offset_cur_nor + data._cur_landmark_normalize[
                    A_feature_closest_landmark_no]
                B_feature_cur_nor = B_feature_closest_landmark_offset_cur_nor + data._cur_landmark_normalize[
                    B_feature_closest_landmark_no]
                # print('B_feature_cur_nor.shape = {}'.format(B_feature_cur_nor.shape))
                A_feature_cur = Utilis.translateTo(A_feature_cur_nor, data._unnor_matrix).astype(int)
                B_feature_cur = Utilis.translateTo(B_feature_cur_nor, data._unnor_matrix).astype(int)
                # print('A_feature_cur = {}, B_feature_cur = {}'.format(A_feature_cur, B_feature_cur))

                # get A, B pixel val via feature cur pos
                image = cv2.imread(data._full_image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                rows = image.shape[0]
                cols = image.shape[1]
                A_feature_pixel = B_feature_pixel = 0
                if A_feature_cur[0, 0] >= 0 and A_feature_cur[0, 0] < cols and A_feature_cur[0, 1] >= 0 and \
                        A_feature_cur[0, 1] < rows:
                    A_feature_pixel = float(image[A_feature_cur[0, 1], A_feature_cur[0, 0]])
                if B_feature_cur[0, 0] >= 0 and B_feature_cur[0, 0] < cols and B_feature_cur[0, 1] >= 0 and \
                        B_feature_cur[0, 1] < rows:
                    B_feature_pixel = float(image[B_feature_cur[0, 1], B_feature_cur[0, 0]])

                threshold = ferm_node_info[Ferm.FEATURE_THRESHOLD]
                # print('A_feature_pixel = {}, B_feature_pixel = {}, (A_feature_pixel - B_feature_pixel) = {}, threshold = {}'.format(A_feature_pixel, B_feature_pixel, (A_feature_pixel - B_feature_pixel), threshold))

                # get pixel diff and split it
                left_node_index = 2 * idx_ferm_node + 1
                right_node_index = 2 * idx_ferm_node + 2
                if (A_feature_pixel - B_feature_pixel) > threshold:
                    data._ferm_node_index = left_node_index
                else:
                    data._ferm_node_index = right_node_index

                # if get score, accumulate and compute mean_left/mean_right to get score, need to reset data._ferm_node_index
                if get_score:
                    if data._ferm_node_index == left_node_index:
                        mean_left += data.getNomalizedLandmarkTruth() - data._cur_landmark_normalize
                        num_left += 1
                    elif data._ferm_node_index == right_node_index:
                        mean_right += data.getNomalizedLandmarkTruth() - data._cur_landmark_normalize
                        num_right += 1
                    data._ferm_node_index = idx_ferm_node
        # if get_score:
        #  print('splitNode get score, iterate all data end')
        # else:
        #  print('splitNode!, iterate all data end')
        if get_score:
            score_left = score_right = 0
            if num_left > 0:
                mean_left /= num_left
                score_left = num_left * Utilis.getDotSelf(mean_left)

            if num_right > 0:
                mean_right /= num_right
                score_right = num_right * Utilis.getDotSelf(mean_right)
            # print('splitNode end, mean_left = {}, num_left = {}, mean_right = {}, num_right = {}'.format(getPartsMatrix(mean_left), num_left, getPartsMatrix(mean_right), num_right))
            return score_left + score_right
        # print('splitNode end!')
        return -1

    def generateFermNodeInfo(self,
                             idx_ferm_node,
                             train_data,
                             feature_pool,
                             feature_closest_landmark_offset,
                             feature_closest_landmark_no):

        # print('generateFermNodeInfo begin, idx_ferm_node = {}'.format(idx_ferm_node))
        # generate candidate feature A and B for node to pick max score one
        candidate_ferm_node_infos = []
        max_score = -1
        idx_max_score = 0
        for idx_candidate in range(self._configuration._num_candidate_ferm_node_infos):
            # print('idx_candidate = {}'.format(idx_candidate))
            feature_A_index = feature_B_index = 0
            prob = prob_threshold = distance = 0.0
            while True:
                feature_A_index = random.randrange(len(feature_pool))
                feature_B_index = random.randrange(len(feature_pool))
                feature_A = feature_pool[feature_A_index]
                feature_B = feature_pool[feature_B_index]
                distance = Utilis.getDistance(feature_A, feature_B)
                prob = math.exp(-distance / self._configuration._lamda)
                prob_threshold = random.uniform(0, 1)
                if feature_A_index != feature_B_index and prob > prob_threshold:
                    break
            MAX_VAL = 255  # uint8 max value
            threshold = (random.random() * float(MAX_VAL) - 128) / 2
            candidate_ferm_node_info = {
                Ferm.A_FEATURE_CLOSEST_LANDMARK_NO: int(feature_closest_landmark_no[feature_A_index]),
                Ferm.B_FEATURE_CLOSEST_LANDMARK_NO: int(feature_closest_landmark_no[feature_B_index]),
                Ferm.A_FEATURE_CLOSEST_LANDMARK_OFFSET_X: feature_closest_landmark_offset[feature_A_index, 0],
                Ferm.A_FEATURE_CLOSEST_LANDMARK_OFFSET_Y: feature_closest_landmark_offset[feature_A_index, 1],
                Ferm.B_FEATURE_CLOSEST_LANDMARK_OFFSET_X: feature_closest_landmark_offset[feature_B_index, 0],
                Ferm.B_FEATURE_CLOSEST_LANDMARK_OFFSET_Y: feature_closest_landmark_offset[feature_B_index, 1],
                Ferm.FEATURE_THRESHOLD: threshold
            }

            # find max score
            score = self.splitNode(idx_ferm_node,
                                   train_data,
                                   candidate_ferm_node_info,
                                   True)
            if score > max_score:
                max_score = score
                idx_max_score = idx_candidate
            # print('candidate_ferm_node_info = {}'.format(candidate_ferm_node_info))
            candidate_ferm_node_infos.append(candidate_ferm_node_info)
            # print('score = {}'.format(score))

        # print('max_score = {}, idx_max_score = {}'.format(max_score, idx_max_score))
        return candidate_ferm_node_infos[idx_max_score]

    def train(self,
              train_data,
              validation_data,
              feature_pool,
              feature_closest_landmark_offset,
              feature_closest_landmark_no):
        print('>>Ferm {} train begin..'.format(self._no))
        for idx_ferm_node in range(len(self._ferm_nodes)):
            # print('before generate, idx_ferm_node = {}, self._ferm_nodes[idx_ferm_node] = {}'.format(idx_ferm_node, self._ferm_nodes[idx_ferm_node]))
            self._ferm_nodes[idx_ferm_node] = self.generateFermNodeInfo(
                idx_ferm_node,
                train_data,
                feature_pool,
                feature_closest_landmark_offset,
                feature_closest_landmark_no)
            # print('after generate, self._ferm_nodes[{}] = {}'.format(idx_ferm_node, self._ferm_nodes[idx_ferm_node]))
            self.splitNode(idx_ferm_node,
                           train_data,
                           self._ferm_nodes[idx_ferm_node],
                           False)
            self.splitNode(idx_ferm_node,
                           validation_data,
                           self._ferm_nodes[idx_ferm_node],
                           False)

        # print('Ferm {} train end, print all train data ferm node index..'.format(self._no))
        # printAllDataFermNodeIndecis(train_data)

        # print('Ferm {} train end, print all valid data ferm node index..'.format(self._no))
        # printAllDataFermNodeIndecis(validation_data)

        # get all residual of leafs
        num_data_leafs = [0] * len(self._ferm_leafs)
        for data in train_data:
            leaf_no = data._ferm_node_index - self.getNodesNum()
            # print('data leaf_no = {}'.format(leaf_no))
            self._ferm_leafs[leaf_no] += data._landmark_truth_normalize - data._cur_landmark_normalize
            # print('data _ferm_leafs[{}] = {}'.format(leaf_no, getPartsMatrix(self._ferm_leafs[leaf_no])))
            num_data_leafs[leaf_no] += 1

        # print('num_data_leafs = {}'.format(num_data_leafs))
        for ferm_leaf_no in range(len(self._ferm_leafs)):
            # print('before, self._ferm_leafs[[{}] = {}'.format(ferm_leaf_no, getPartsMatrix(self._ferm_leafs[ferm_leaf_no])))
            if num_data_leafs[ferm_leaf_no] > 0:
                self._ferm_leafs[ferm_leaf_no] /= num_data_leafs[ferm_leaf_no]
            # print('after, self._ferm_leafs[[{}] = {}'.format(ferm_leaf_no, getPartsMatrix(self._ferm_leafs[ferm_leaf_no])))

    def predict(self, datas):
        for idx_ferm_node in range(len(self._ferm_nodes)):
            self.splitNodeForPredict(idx_ferm_node,
                           datas,
                           self._ferm_nodes[idx_ferm_node])

    FERM_NAME = 'ferm_name'
    FERM_NAME_VAL = 'ferm'
    FERM_NODES = 'ferm_nodes'
    FERM_LEAFS = 'ferm_leafs'

    def getJSONObj(self):
        result = {}

        # ferm name, might be ignore
        ferm_name_val = '{}_{}'.format(Ferm.FERM_NAME_VAL, self._no)
        result[Ferm.FERM_NAME] = ferm_name_val

        # ferm nodes
        ferm_nodes_val = []
        for ferm_node_info in self._ferm_nodes:
            ferm_nodes_val.append(ferm_node_info)

        # ferm_leafs
        ferm_leafs_val = []
        for ferm_leaf in self._ferm_leafs:
            ferm_leafs_val.append(ferm_leaf.tolist())

        result[Ferm.FERM_NODES] = ferm_nodes_val
        result[Ferm.FERM_LEAFS] = ferm_leafs_val
        return result

    def fromJSONOBJ(self, json_obj):
        # print('ferm json_obj = {}'.format(json_obj))

        # ferm nodes
        for idx_ferm_node_info in range(len(json_obj[Ferm.FERM_NODES])):
            self._ferm_nodes[idx_ferm_node_info] = json_obj[Ferm.FERM_NODES][idx_ferm_node_info]

        # ferm leafs
        for idx_ferm_leaf in range(len(json_obj[Ferm.FERM_LEAFS])):
            self._ferm_leafs[idx_ferm_leaf] = np.asarray(json_obj[Ferm.FERM_LEAFS][idx_ferm_leaf])
            # print('self._ferm_leafs[{}].shape = {}'.format(idx_ferm_leaf, self._ferm_leafs[idx_ferm_leaf].shape))
            # print('self._ferm_leafs[{}] = {}'.format(idx_ferm_leaf, getPartsMatrix(self._ferm_leafs[idx_ferm_leaf])))
        return