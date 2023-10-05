import datetime

from data.preprocess import *
from data.data_params import *

def main():
    # prepare the dataset
    print("\n\n\n============================================================")
    print("Starting download the dataset.")
    print(datetime.datetime.now())
    print("============================================================")
    if not os.path.exists(dataset_path):
        if download_file():
            unzip_file()
            print('dataset prepared successfully!')
        else:
            print('dataset prepared failed!')
    else:
        print('dataset prepared successfully!')
        
    # preprocess the data for training
    print("\n\n\n============================================================")
    print("Starting preprocess the dataset.")
    print(datetime.datetime.now())
    print("============================================================")
    if not os.path.exists(train_list_path):
        data2csvList("train")
    if not os.path.exists(test_list_path):
        data2csvList("test")
        
if __name__=="__main__":
    main()