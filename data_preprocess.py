import datetime

from data.preprocess import *
from data.data_params import *

def main():
    # prepare the dataset
    print("============================================================")
    print("Start downloading the dataset.", datetime.datetime.now())
    if not os.path.exists(dataset_path):
        if download_file():
            unzip_file()
            print('dataset prepared successfully!')
        else:
            print('dataset prepared failed!')
            return
    else:
        print('dataset prepared successfully!', datetime.datetime.now())
    print("============================================================")
        
    # preprocess the data for training
    print("\n\n============================================================")
    print("Start preprocessing the dataset.", datetime.datetime.now())
    if not os.path.exists(train_list_path):
        data2csvList("train")
    if not os.path.exists(test_list_path):
        data2csvList("test")
    print("Dataset have been proprecessed!", datetime.datetime.now())
    print("============================================================")
        
    # extract mfcc feature
    print("\n\n============================================================")
    print("Starting extracting the mfcc feature.", datetime.datetime.now())
    if not os.path.exists(train_mfcc_list):
        mfccExtract("train")
    if not os.path.exists(test_mfcc_list):
        mfccExtract("test")
    print("mfcc features have been extracted!", datetime.datetime.now())
    print("============================================================")
    
    print("\n\nDone!")
        
if __name__=="__main__":
    main()