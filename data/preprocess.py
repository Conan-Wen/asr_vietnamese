import requests
from tqdm import tqdm
import tarfile
import glob
import csv
import pandas as pd
import tensorflow as tf
import librosa

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
        prompts_path = train_prompts_path
        wavs_path = train_wavs_path
        csv_path = train_list_path
    elif dataSet=="test":
        prompts_path = test_prompts_path
        wavs_path = test_wavs_path
        csv_path = test_list_path
    else:
        raise ValueError("dataSet parameter must be train or test!")
    
    with open(prompts_path, encoding="utf8", mode="r") as rf, \
            open(csv_path, mode="w") as wf:
        data = rf.read()
        lines = data.split("\n")
        
        csv_writer = csv.writer(wf)
        csv_writer.writerow(["data_name", "file_path", "transcription"])
        
        for line in lines:
            instance = line.split(" ", 1)
            if len(instance) == 1:
                continue
            file_path = glob.glob(os.path.join(wavs_path, "**", instance[0]+".wav"))[0]
            transcription = instance[1]
            csv_writer.writerow([instance[0], file_path, transcription])

# compute and save mfcc feature
def mfccExtract(dataSet="train"):
    if dataSet=="train":
        list_path = train_list_path
        mfcc_path = train_mfcc_path
        mfcc_list = train_mfcc_list
    elif dataSet=="test":
        list_path = test_list_path
        mfcc_path = test_mfcc_path
        mfcc_list = test_mfcc_list
    else:
        raise ValueError("dataSet parameter must be train or test!")
    
    df = pd.read_csv(list_path)
    
    num_spectrogram_bins = int(fft_length / 2) + 1
    
    mel_filter_bank = tf.signal.linear_to_mel_weight_matrix(    
      num_mel_bins,
      num_spectrogram_bins,
      sampling_rate,
      lower_edge_hertz,
      upper_edge_hertz,
    )
    
    os.makedirs(mfcc_path, exist_ok=True)
    with open(mfcc_list, "w") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["data_name", "file_path", "transcription"])
        
        for index, row in tqdm(df.iterrows(), total=len(df)):
            data_name, wav_file_path, transcription = row["data_name"], row["file_path"], row["transcription"]
            
            # read wave data, and downsampling
            wav_file = tf.io.read_file(wav_file_path)
            waveform, sr = tf.audio.decode_wav(wav_file)
            waveform = tf.squeeze(waveform, axis=-1)
            waveform = tf.cast(waveform, tf.float32)
            waveform = librosa.resample(waveform.numpy(), orig_sr=sr, target_sr=sampling_rate)
            
            spectrogram = tf.signal.stft(
                waveform,
                frame_length=frame_length,
                frame_step=frame_step,
                fft_length=fft_length
            )

            spectrogram = tf.abs(spectrogram)
            pow_spectrogram = tf.math.pow(spectrogram, 2)
            
            mel_spectrogram = tf.tensordot(pow_spectrogram, mel_filter_bank, 1)
            mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(mel_spectrogram.shape[-1:]))
            
            log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

            # normalization
            means = tf.math.reduce_mean(log_mel_spectrogram, 1, keepdims=True)
            stddevs = tf.math.reduce_std(log_mel_spectrogram, 1, keepdims=True)
            log_mel_spectrogram = (log_mel_spectrogram - means) / (stddevs + 1e-10)
            
            out_file = os.path.join(mfcc_path, data_name) + ".bin"
            log_mel_spectrogram.numpy().tofile(out_file)
            
            csv_writer.writerow([data_name, out_file, transcription])