import requests
from tqdm import tqdm
import tarfile
import glob
import csv
import pandas as pd

from data.data_params import *


# download the vivos dataset
def download_file():
    if os.path.exists(tar_file_path):
        return True
    
    try:
        response = requests.get(data_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)
        
        os.makedirs(folder_path, exist_ok=True)
        with open(tar_file_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)

        progress_bar.close()
    except Exception as e:
        print(e)
        os.remove(folder_path)
        return False
    else:
        if total_size != 0 and progress_bar.n != total_size:
            return False
        return True

# unzip the dataset
def unzip_file():
    try:
        with tarfile.open(tar_file_path, 'r:gz') as tar:
            tar.extractall(path=folder_path)
    except Exception as e:
        print(e)
    else:
        os.remove(tar_file_path)

# preprocess the data, and save them to csv file to load as DataForm later
def data2csvList(dataSet="train"):
    if dataSet=="train":
        data_path = train_data_path
        prompts_path = train_prompts_path
        wavs_path = train_wavs_path
    elif dataSet=="test":
        data_path = test_data_path
        prompts_path = test_prompts_path
        wavs_path = test_wavs_path
    else:
        raise ValueError("dataSet parameter must be train or test!")
    
    with open(prompts_path, encoding="utf8", mode="r") as rf, \
            open(os.path.join(data_path, "dataList.csv"), mode="w") as wf:
        data = rf.read()
        lines = data.split("\n")
        
        csv_writer = csv.writer(wf)
        csv_writer.writerow(["file_path", "transcription"])
        
        for line in lines:
            instance = line.split(" ", 1)
            if len(instance) == 1:
                continue
            file_path = glob.glob(os.path.join(wavs_path, "**", instance[0]+".wav"))[0]
            transcription = instance[1]
            csv_writer.writerow([file_path, transcription])
            
if __name__=="__main__":
    data2csvList("test")
    tmp = pd.read_csv(train_list_path)
    print(tmp.head(3))
    
            
