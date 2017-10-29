import numpy as np
import pandas
import matplotlib.pyplot as plt
from os.path import isfile, join
from os import listdir

def rename_columns(df, index_name, prefix_name):
    l = ['{}_{}'.format(prefix_name, c) for c in df.columns]
    l[0] = index_name
    df.columns = l
    return df

def get_files_dataframes(path):
    columns=['audio', 'eyes', 'face_nn', 'kinect', 'labels']
    df = pandas.DataFrame(np.zeros((0, len(columns)), int), columns=columns)
    #print (df.head())
    for index, dir_name in enumerate(columns):
        current_path = path + dir_name
        onlyfiles = [f for f in listdir(current_path) if isfile(join(current_path, f))]
        for file in onlyfiles:
            df.loc[file, dir_name] = 1
    return df.fillna(0)

def round_features_time(featuresList):
    for f in featuresList:
        f['Time'] = round(f['Time'], 2)

def merge_all_features(featureList):
    res = featureList[0]
    for f in featureList[1:]:
        res = res.merge(f, how="outer", on='Time')
    return res.sort_values('Time')

def get_features_for_file_name(fileName):
    audio = pandas.read_csv('data/train/audio/{}.csv'.format(fileName))
    
    eyes = pandas.read_csv('data/train/eyes/{}.csv'.format(fileName), skiprows=[0], header=None)
    eyes = rename_columns(eyes, 'Time', 'eyes')
    
    face_nn = pandas.read_csv('data/train/face_nn/{}.csv'.format(fileName), skiprows=[0], header=None)
    face_nn = rename_columns(face_nn, 'Time', 'face')
    
    kinect = pandas.read_csv('data/train/kinect/{}.csv'.format(fileName))
    
    labels = pandas.read_csv('data/train/labels/{}.csv'.format(fileName))
    
    round_features_time([labels, audio, eyes, face_nn, kinect])
    
    res = merge_all_features([labels, audio, eyes, face_nn, kinect])
    
    return res.fillna(0)

def plot(x, y, xlabel, ylabel):
    plt.plot(x, y)
    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    
train = get_features_for_file_name('id124bf1d0')
#plot(labels['Time'], labels['Agreement score'], 't', 'arg scr')

files_df = get_files_dataframes('data/train/')
files_df.head()
files_df_exists = files_df[(files_df['audio'] == 1) & (files_df['eyes'] == 1) & (files_df['face_nn'] == 1) & (files_df['kinect'] == 1)]
indexes = files_df_exists.index
