class Configuration:
  def __init__(self,
               num_landmarks,
               train_data_times,
               cascade_number,
               ferm_number,
               ferm_group_number,
               ferm_depth,
               num_candidate_ferm_node_infos,
               feature_pool_size,
               shrinkage_factor,
               padding,
               lamda):
    self._num_landmarks = num_landmarks
    self._train_data_times = train_data_times
    self._cascade_number = cascade_number
    self._ferm_number = ferm_number
    self._ferm_num_per_group = ferm_group_number
    self._ferm_depth = ferm_depth
    self._num_candidate_ferm_node_infos = num_candidate_ferm_node_infos
    self._feature_pool_size = feature_pool_size
    self._shrinkage_factor = shrinkage_factor
    self._padding = padding
    self._lamda = lamda