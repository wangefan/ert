import Utilis
from Configuration import Configuration
from ERT import ERT

root = r"./lfpw"
train_path = root + r"/trainset_small"
test_path = root + r"/testset_small"
train_images_path = train_path + r"/images"
train_labels_path = train_path + r"/labels"
test_images_path = test_path + r"/images"
test_labels_path = test_path + r"/labels"
model_full_path = root + r"/ert_model_small.json"

# 1.Prepare data
print('1. Prepare data')
train_data = []
Utilis.load_data(train_images_path, train_labels_path, train_data)
validation_data = []
Utilis.load_data(test_images_path, test_labels_path, validation_data)

# 2.configuratuin and process datas
print('2. Configurate and process data')
num_landmarks = int(train_data[0].getLandmarkTruth().shape[0])
train_data_times = 2
cascade_number = 10
ferm_number = 5
ferm_group_number = 1
ferm_depth = 5
num_candidate_ferm_node_infos = 15
feature_pool_size = 400
shrinkage_factor = 0.1
padding = 0.1
lamda = 0.1
configuration = Configuration(num_landmarks,
                              train_data_times,
                              cascade_number,
                              ferm_number,
                              ferm_group_number,
                              ferm_depth,
                              num_candidate_ferm_node_infos,
                              feature_pool_size,
                              shrinkage_factor,
                              padding,
                              lamda)
ert = ERT(configuration)

ert.processData(train_data, validation_data)

validation_data[2].showCurLandmarks()

# 3.train
print('3. Begin to train..')
ert.train()

# 4. ert.save
print('4. Train done, save model..')
ert.save(model_full_path)

# 5. finish
print('5. Finish!')