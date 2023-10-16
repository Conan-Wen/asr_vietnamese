# asr_vietnamese

## Step
### Data Preprocessing
First, excuted the `data_process.py` to preprocess the data.

## Step explanation
### Data Preprocessing
`data_process.py` will download the dataset, and generate a csv file for DataFrame to load the data.

Then, extracts the MFCC feature for training by the csv file generatedd before.

The MFCC features will be saved as tf.float32 Tensor, and stored as binary file.