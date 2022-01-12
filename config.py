import os
import torch

data_dir = os.getcwd() + '/data/'
train_dir = data_dir + 'train.npz'
test_dir = data_dir + 'test.npz'
predict_dir = data_dir + 'predict.npz'
files = ['train', 'test', 'predict']
bert_model = '../roberta_wwm/'
roberta_model = '../roberta_wwm/'
# model_dir = os.getcwd() + '/save_model/'
model_dir = os.getcwd() + '/save_model/'
log_dir = model_dir + 'train.log'
case_dir = os.getcwd() + '/case/bad_case.txt'
predict_seq_output_dir = os.getcwd() + '/predict/s_seq_output.txt'
predict_dict_output_dir = os.getcwd() + '/predict/s_dict_output.txt'
predict_entity_list_dir = os.getcwd() + '/predict/s_entity_list_output.txt'

# 训练集、验证集划分比例
dev_split_size = 0.1

# 是否加载训练好的NER模型
load_before = False

# 是否对整个BERT进行fine tuning
full_fine_tuning = True

# hyper-parameter
learning_rate = 3e-5
weight_decay = 0.01
clip_grad = 5

batch_size = 8
epoch_num = 50
min_epoch_num = 5
patience = 0.0002
patience_num = 10

gpu = '7'

if gpu != '':
    device = torch.device(f"cuda:{gpu}")
else:
    device = torch.device("cpu")

labels = ['element', 'attribute']

label2id = {
    "O": 0,
    "B-element": 1,
    "B-attribute": 2,
    "I-element": 3,
    "I-attribute": 4,
    "S-element": 5,
}

id2label = {_id: _label for _label, _id in list(label2id.items())}
