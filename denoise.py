# Imports
import os, sys, socket
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

#Datasource
DATAFOLDER = 'Lina2'
# Constants
# Have only verified 50, 100, 250
BATCH_SIZE = 100
# Have only verified 512, 1024, 2048, 4096
WINDOW_SIZE = 4096

#0 is accept records that are all one state; 0.5 is,accept only records with more than 50% of each state
#Not sure what happens with 0.5 or more so recommend range of 0 to 0.25
#Note if too high preprocess data will loop indefinitely at some stage.
THRESHOLD =0.05

#Preprocessed method -- clean signal generator using https://github.com/RichardBJ/DeepGANnel
def check_window(channel_record: np.ndarray) -> bool:
    
    """ Returns True if there is an event in the window """
    
    """return True if (1 in channel_record[:,1:]) and (0 in channel_record[:,1:]) else False"""
    """toughen this up, want more than one single event :-)"""
    threshold = THRESHOLD #0.05
    total=len(channel_record[:,1])
    opens=len(channel_record[channel_record[:,1]==1])
    val=True if ((opens/total >threshold) and (opens/total <(1-threshold))) else False
    return val
    
def preprocess_data():
    
    """ 
    Preprocessing data into one large numpy array - along with a list of
    marker start indexes for randomly sampling batches.
    
    If an index < marker AND index + WINDOW_SIZE > marker, they'll be a change
    of file in the window, therefore it's invalid.
    """
    
    cwd = os.getcwd()
    files = glob.glob(f'{DATAFOLDER}/*.csv')
    outer_index = 1
    markers = []
    output_data = np.array([[0,0]])
    for filename in files:
        print(f'Processing {filename}')
        """currently starting from row 2 incase headers etc"""
        inner_index = 0
        active = False
        timecol=False
        if timecol==True:
            data = pd.read_csv(filename).values[2:,1:]
        else:
            data = pd.read_csv(filename).values[2:,:]
        '''round idealisation to 0 and 1
        this is typically, but not always necessary depending on record source'''
        data[:,1]=np.round(data[:,1])
        data[:,1]=data[:,1]/np.max(data[:,1])
        data[:,1]=np.round(data[:,1]) # normalisation on the second row
        for i in range(len(data)):
            window_check = check_window(data[i:i+WINDOW_SIZE, :])
            markers.append(window_check)
            if window_check and not active:
                # Start recording - make a note of the inner index.
                inner_index = i
                active = True
            elif not window_check and active:
                # Stop "recording" and save the file to the output data.
                # Also make a marker for safe indexing later.
                window_to_save = data[inner_index: i + WINDOW_SIZE - 1, :]
                end_index = outer_index + i + WINDOW_SIZE - 1
                
                output_data = np.concatenate((output_data, window_to_save))
                """markers.append(end_index)"""
                               
                active = False
        outer_index = len(output_data)
    return (data, markers)


def segment(seg, s):
    """
    Segment the signal into batch of sequence size (4096)
    Args:
        seg (_type_): the original sequence
        s (_type_): 4096

    Returns:
        _type_: sequence is split into (seq // s + 1, s)
    """
    seg = np.concatenate((seg, np.zeros((-len(seg)%s))))
    nrows=((seg.size-s)//s)+1
    n=seg.strides[0]
    return np.lib.stride_tricks.as_strided(seg, shape=(nrows,s), strides=(s*n, n))

def slide_window(seq, s):
    """
    slide the window size of s from start to end
    Args:
        seg (_type_): the original sequence
        s (_type_): 4096

    Returns:
        _type_: sequence is split into (seq - s + 1, s)
    """
    v = np.lib.stride_tricks.sliding_window_view(seq, s)
    return v
    
def add_noise(seq, mu, sigma):
    """
    Add noise to the given signal sequence
    Args:
        seq (_type_): the orignal signal with size (seq // s + 1, s)
        mu (_type_): mean of gaussian distribution
        sigma (_type_): variance of gaussian distribution

    Returns:
        noisy signal sequence with size (seq // s + 1, s)
    """
    noise = np.random.normal(mu,sigma, seq.shape)
    seq = seq + noise
    return seq


clean_data, _ = preprocess_data()
clean_data = slide_window(clean_data[:,0], 4096)
noisy_data = add_noise(clean_data, mu = 0, sigma = 0.1)
# notice that the noisy_data is a (790000 - 4096 + 1, 4096) np array, thus it is huge
row = noisy_data.shape[0] # calculate the length of sequence 
idx = np.arange(row)
np.random.shuffle(idx)
train, test, valid = noisy_data[:0.6*row], noisy_data[0.6*row :0.8*row], noisy_data[0.8*row:]
# draw the graph compared with clean data and noisy data    
# import matplotlib.pyplot as plt
# f = plt.figure(figsize=(10,6))
# ax = f.add_subplot(121)
# ax2 = f.add_subplot(122)
# ax.plot(clean_data[0])
# ax.title.set_text("The clean signal")
# ax2.plot(noisy_data[0])
# ax2.title.set_text(r"The noisy signal with $\backsim\mathcal{N}(0,0.1)$")
# plt.savefig('noise.pdf')
# plt.show()


