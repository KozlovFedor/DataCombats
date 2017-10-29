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
    audio = pandas.read_csv('data/train/audio/{}'.format(fileName))
    
    eyes = pandas.read_csv('data/train/eyes/{}'.format(fileName), skiprows=[0], header=None)
    eyes = rename_columns(eyes, 'Time', 'eyes')
    
    face_nn = pandas.read_csv('data/train/face_nn/{}'.format(fileName), skiprows=[0], header=None)
    face_nn = rename_columns(face_nn, 'Time', 'face')
    
    kinect = pandas.read_csv('data/train/kinect/{}'.format(fileName))
    
    labels = pandas.read_csv('data/train/labels/{}'.format(fileName))
    
    round_features_time([labels, audio, eyes, face_nn, kinect])
    
    res = merge_all_features([labels, audio, eyes, face_nn, kinect])
    
    return res.fillna(0)

def plot(x, y, xlabel, ylabel):
    plt.plot(x, y)
    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    
def plot_pred(x, y, y_pred, xlabel, ylabel):
    plt.plot(x, y)
    plt.plot(x, y_pred)
    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    

#plot(labels['Time'], labels['Agreement score'], 't', 'arg scr')

files_df = get_files_dataframes('data/train/')
#files_df.head()
files_df_exists = files_df[(files_df['audio'] == 1) & (files_df['eyes'] == 1) & (files_df['face_nn'] == 1) & (files_df['kinect'] == 1)]
#indexes = files_df_exists.index

Train_features = pandas.DataFrame()
Test_features = pandas.DataFrame()
ind = 0
for id in files_df_exists.index:
    features = get_features_for_file_name(id)
    features = features.drop(['Time'], axis=1)
    features['id'] = ind
    features = features.iloc[::7,:]
    if ind < 100:
        if ind == 0:
            Train_features = features
        else:
            Train_features = Train_features.append(features)
    elif ind < 120:
        if ind == 100:
            Test_features = features
        else:
            Test_features = Test_features.append(features)
    else:
        break
    print(ind)
    ind += 1

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

cv = KFold(n_splits=5, shuffle=True)

Y_train = Train_features.loc[:,'Anger':'Neutral']
Train_features = Train_features.loc[:, 'Agreement score':]

X = np.array(Train_features)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = {}

for emotion in Y_train:
    clf[emotion] = LogisticRegression()
    #X_train, X_test, y_train, y_test = train_test_split(X_scaled, np.array(Y[emoution]))
    #clf.fit(X_train, y_train)
    score = cross_val_score(clf[emotion], X_scaled, np.array(Y_train[emotion]), scoring='roc_auc', cv=cv)
    #pred = clf.predict_proba(X_test)[:, 1]
    print(emotion, score.mean())#roc_auc_score(y_test, pred))

Y_test = Test_features.loc[:,'Anger':'Neutral']
Test_features = Test_features.loc[:, 'Agreement score':]

X_test = np.array(Test_features)
X_test_scaled = scaler.fit_transform(X_test)
pred_emotions = {}
for emotion in Y_test:
    clf = LogisticRegression(C=0.1)
    clf.fit(X_scaled, np.array(Y_train[emotion]))
    pred_emotions[emotion] = clf.predict_proba(X_test_scaled)[:, 1]
    print(emotion, roc_auc_score(Y_test[emotion], pred_emotions[emotion]))
'''
x_range = range(0, len(Y_test))
for emotion in Y_test:
    plot_pred(x_range, Y_test.loc[:, emotion], pred_emotions[emotion] , "index", "result_" + emotion)'''