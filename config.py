# dataset name 
dataset = 'ml-1m'
assert dataset in ['ml-1m', 'pinterest-20']

# model name 
model = 'NeuMF-end'
assert model in ['MLP', 'GMF', 'NeuMF-end', 'NeuMF-pre']

# paths
main_path = './Data/'

# train_rating = main_path + '{}.train.rating'.format(dataset)
train_rating = main_path + 'Tianchi_Train_uid_iid_rt.csv'
# test_rating = main_path + '{}.test.rating'.format(dataset)
test_rating = main_path + 'Tianchi_Test_uid_iid.csv'
# test_negative = main_path + '{}.test.negative'.format(dataset)
test_negative = main_path + 'Tianchi_Test_uid_iid.csv'

model_path = './models/'
GMF_model_path = model_path + 'GMF.pth'
MLP_model_path = model_path + 'MLP.pth'
NeuMF_model_path = model_path + 'NeuMF.pth'
