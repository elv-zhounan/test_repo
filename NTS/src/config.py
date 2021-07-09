# BATCH_SIZE = 60 # can be changed depending on memory of GPU
# PROPOSAL_NUM = 4 # fixed hyperparameter
# CAT_NUM = 2 # fixed hyperparameter
# INPUT_SIZE = (448, 448)  # (w, h), fixed hyperparameter
# LR = 0.001
# WD = 1e-4
# SAVE_FREQ = 1
# resume = ''
# data_root='./data'
# test_model = './model_data/model.ckpt'
# save_dir = './nts_training_logs/'


#NUM_CLASSES = 1145
NUM_CLASSES = 2341  # the Logo-2k+ dataset
#NUM_CLASSES = 1369
NUM_POOL_CLASSES = 1377
BATCH_SIZE = 10
PROPOSAL_NUM = 6
CAT_NUM = 4
INPUT_SIZE = (448, 448)  # (w, h)
LR = 0.0005
WD = 1e-4
SAVE_FREQ = 1
resume = ''#'model_data/1369class.ckpt'
data_root='./data'
test_model = 'model_data/model_ResNet152_Top6_CAT4.ckpt'
save_dir = './nts_training_logs/'