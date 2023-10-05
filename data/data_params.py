import os


## Parameters for data download
# vivos datasets url to download
data_url = "http://ailab.hcmus.edu.vn/assets/vivos.tar.gz"

# Current Dirctory
cur_dir = os.path.dirname(os.path.abspath(__file__))

# folder path
folder_path = os.path.join(cur_dir, "datasets")

# file path
tar_file_path = os.path.join(cur_dir, "datasets", "vivos.tar.gz")

# path of save datasets
dataset_path = os.path.join(cur_dir, "datasets", "vivos")


## Parameters for data preprocessing
# Trainning data path, and test data path
train_data_path = os.path.join(dataset_path, "train")
test_data_path = os.path.join(dataset_path, "test")

# the prompts.txt path of dataset
train_prompts_path = os.path.join(train_data_path, "prompts.txt")
test_prompts_path = os.path.join(test_data_path, "prompts.txt")

# the wave files path of dataset
train_wavs_path = os.path.join(train_data_path, "waves")
test_wavs_path = os.path.join(test_data_path, "waves")

# the path of datalist in csv file
train_list_path = os.path.join(train_data_path, "dataList.csv")
test_list_path = os.path.join(test_data_path, "dataList.csv")
