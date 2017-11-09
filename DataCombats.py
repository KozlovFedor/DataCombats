import numpy as np
import pandas
import matplotlib.pyplot as plt
from os.path import isfile, join
from os import listdir
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score
import datetime

RANDOM_SEED = 42

def get_all_exists_dataframes(files_df):
    """
        Возвращаем таблицу, где для для файлов(files_df) присутствуют все группы признаков (audio, eyes, face_nn, kinect) 
    """
    files_df_all_exists = files_df[(files_df['audio'] == 1) & (files_df['eyes'] == 1) & (files_df['face_nn'] == 1) & (files_df['kinect'] == 1)]
    return files_df_all_exists

def rename_columns(df, index_name, prefix_name):
    """
        Возвращает dataframe с переименованными столбцами с добавлением префикса prefix_name к исходному имени столбцу.
        При этом считается, что в исходном dataframe первый столбец с именем 'Time'
    """
    l = ['{}_{}'.format(prefix_name, c) for c in df.columns]
    l[0] = index_name
    df.columns = l
    return df

def get_template_dataframe(directory_path, files_df, data_type_names):
    """
        Возвращает общий шаблон для dataframe по файлам(files_df) на основе групп признаков(data_type_names), лежащих в каталоге directory_path.
        Пример параметров:
        directory_path = 'data/train/'
        data_type_names = ['audio', 'eyes', 'face_nn', 'kinect', 'labels']
    """
    files_all_exists_df = get_all_exists_dataframes(files_df)
    file_all_exists = files_all_exists_df.iloc[0]
    columns_features = ['Time']
    for data_type in data_type_names:
        df_current = get_feature_for_name(directory_path, data_type, file_all_exists.name)
        columns_features.extend(df_current.columns[1:])
    return pandas.DataFrame(columns=columns_features)

def get_files_dataframes(path, data_type_names):
    """
        Возвращаем таблицу со значениями существования группы признаков(столбцы data_type_names) для имени файла(строки) по заданному пути path
        Пример параметров:
        path = 'data/train/'
        data_type_names = ['audio', 'eyes', 'face_nn', 'kinect', 'labels']
    """
    df = pandas.DataFrame(np.zeros((0, len(data_type_names)), int), columns=data_type_names)
    #print (df.head())
    for index, dir_name in enumerate(data_type_names):
        current_path = path + dir_name
        onlyfiles = [f for f in listdir(current_path) if isfile(join(current_path, f))]
        for file in onlyfiles:
            df.loc[file, dir_name] = 1
    return df.fillna(0)

def round_features_time(featuresList):
    """
        Округление признака 'Time' до двух знаков после запятой
    """
    for f in featuresList:
        f['Time'] = round(f['Time'], 2)

def merge_all_features(featureList, template_df):
    """
        Объединяет dataframe из списка признаков featureList по заданному шаблону. Шаблон не обязательно должен быть задан.
    """
    res = pandas.DataFrame()       
    if template_df is not None:
        # если был передан шаблон
        res = pandas.DataFrame(np.empty((len(featureList[0].index), len(template_df.columns))).fill(np.nan), columns=template_df.columns)        
        res['Time'] = featureList[0]['Time']
        for df in featureList:
            curr_columns = [column for column in df.columns if column != 'Time']
            res.loc[res['Time'].isin(df['Time']), curr_columns] =  df[curr_columns].values #curr_df.loc[curr_df['Time'].isin(test_df['Time']), curr_columns].values            
    else:
        # если шаблон не был передан
        res = featureList[0]
        for df in featureList[1:]:
            res = res.merge(df, how="outer", on='Time')
        res = res.sort_values('Time')    
    return res

def read_file_csv(file_path, skiprows = False, prefix_rename = 'feature'):
    """
        Считывает .csv файл по заданному пути file_path.
    """
    result = None
    if isfile(file_path):
        if (skiprows):
            result = pandas.read_csv(file_path, skiprows=[0], header=None)
            result = rename_columns(result, 'Time', prefix_rename)
        else:
            result = pandas.read_csv(file_path)
    return result

def get_features_for_file_name(directory_path, file_name, template_df = None):
    """
        Получение общего dataframe всех групп признаков в каталоге directory_path для файла file_name.
    """
    result_features_list = []  
    # labels
    labels_file = '{}labels/{}'.format(directory_path, file_name)
    labels = read_file_csv(labels_file)
    if labels is not None:
        result_features_list.append(labels)
    # audio
    audio_file = '{}audio/{}'.format(directory_path, file_name)
    audio = read_file_csv(audio_file)
    if audio is not None:
        result_features_list.append(audio)    
    # eyes
    eyes_file = '{}eyes/{}'.format(directory_path, file_name)
    eyes = read_file_csv(eyes_file, True, 'eyes' )
    if eyes is not None:
        result_features_list.append(eyes)  
    # face_nn
    face_nn_file = '{}face_nn/{}'.format(directory_path, file_name)
    face_nn = read_file_csv(face_nn_file, True, 'face')
    if face_nn is not None:
        result_features_list.append(face_nn)  
    # kinect
    kinect_file = '{}kinect/{}'.format(directory_path, file_name)
    kinect = read_file_csv(kinect_file)
    if kinect is not None:
        result_features_list.append(kinect)      
    round_features_time(result_features_list) #округляем по время
    res = merge_all_features(result_features_list, template_df) # объединяем группы признаков в один dataframe
    
    # заполняем столбец признаков с пропусками NaN методом усреднения пропущенных значений между соседними
    # https://stackoverflow.com/questions/44102794/imput-missed-values-with-mean-of-nearest-neighbors-in-column
    for col in res:
        res[col] = (res[col].fillna(method='ffill') + res[col].fillna(method='bfill')) / 2 
    
    return res.fillna(0) # возвращаем с заполнение нулями пропусков

def get_feature_for_name(directory_path, data_type, file_name):
    """
        Получение dataframe по заданному имени файла(file_name) для группы признаков (data_type) в каталоге(directory_path)
    """
    if (data_type == 'eyes'):
        eyes = pandas.read_csv('data/train/eyes/{}'.format(file_name), skiprows=[0], header=None)
        eyes = rename_columns(eyes, 'Time', 'eyes')
        return eyes
    elif (data_type == 'face_nn'):
        face_nn = pandas.read_csv('data/train/face_nn/{}'.format(file_name), skiprows=[0], header=None)
        face_nn = rename_columns(face_nn, 'Time', 'face')
        return face_nn
    else:
        return pandas.read_csv('data/train/{}/{}'.format(data_type, file_name))
    
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

def make_id_from_filename(file_name):
    """
        Convert file name to number. Assuming that file name has format: 'id*.csv'
    """
    return int(file_name[2:-4], 16)

def make_features_df_for_ids(ids, agreement_score=None, verbose=False):
    """
        Makes dataframe from list of @ids using only rows, where Agreement score >= @agreement_score
    """
    if verbose:
        print('Start make_features_df_for_ids. Total files to process:', len(ids))
    start_time = datetime.datetime.now()
    res = pandas.DataFrame()
    ind = 0
    for file_name in ids:
        # Получаем все признаки в куче
        features = get_features_for_file_name(directory_path, file_name, template_df)
        features = features.drop(['Time'], axis=1)
        #features['id'] = make_id_from_filename(file_name)
        #features = features.iloc[::7,:]
        if agreement_score:
            features = features[features['Agreement score'] >= agreement_score]
        if ind > 0:
            res = res.append(features)
        else:
            res = features
        ind += 1
    if verbose:
        print('Done. Total time:', datetime.datetime.now() - start_time)
    return res

data_type_names= ['audio', 'eyes', 'face_nn', 'kinect', 'labels']
x_type_names = ['audio', 'eyes', 'face_nn', 'kinect']

#plot(labels['Time'], labels['Agreement score'], 't', 'arg scr')

# Получаем информацию о всех обрабатываемых файлах
directory_path = 'data/train/'
files_df = get_files_dataframes(directory_path, data_type_names)

# Выбираем файлы для дальнейшей обработки
#files_to_process = files_df # выбираем все файлы, даже с пропуском групп признаков
files_to_process = get_all_exists_dataframes(files_df) # выбираем файлы без пропусков групп признаков

# Формируем шаблон для всех признаков на основе перечисленных в списке по группам признаков(data_type_names)
template_df = get_template_dataframe(directory_path, files_df, data_type_names)

# Формируем тестовую и тренировочную выборки

Train_ids, Test_ids = train_test_split(files_df.index[:50], random_state=RANDOM_SEED)

Train_features = make_features_df_for_ids(Train_ids, verbose=True)
Test_features = make_features_df_for_ids(Test_ids, verbose=True)

#cv = KFold(n_splits=5, shuffle=True)
columns_objects = ['Anger', 'Sad', 'Disgust', 'Happy', 'Scared', 'Neutral']
columns_features = [col for col in Train_features.columns if col not in columns_objects]

Y_train = Train_features[columns_objects]
Train_features = Train_features[columns_features].drop(['Agreement score'], axis = 1)

X = np.array(Train_features)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
'''
for emotion in Y_train:
    start_time = datetime.datetime.now()
    clf = LogisticRegression()
    #X_train, X_test, y_train, y_test = train_test_split(X_scaled, np.array(Y[emoution]))
    #clf.fit(X_train, y_train)
    score = cross_val_score(clf, X_scaled, np.array(Y_train[emotion]), scoring='roc_auc', cv=cv)
    #pred = clf.predict_proba(X_test)[:, 1]
    print(emotion, score.mean(), 'Time:', datetime.datetime.now() - start_time)#roc_auc_score(y_test, pred))'''

Y_test = Test_features[columns_objects]
Test_features_agreement = Test_features.loc[:, 'Agreement score']
Test_features = Test_features[columns_features].drop(['Agreement score'], axis = 1)

X_test = np.array(Test_features)
X_test_scaled = scaler.fit_transform(X_test)
pred_emotions = {}
'''
for emotion in Y_test:
    start_time = datetime.datetime.now()
    clf = LogisticRegression()
    clf.fit(X_scaled, np.array(Y_train[emotion]))
    pred_emotions[emotion] = clf.predict_proba(X_test_scaled)[:, 1]
    print(emotion, roc_auc_score(Y_test[emotion], pred_emotions[emotion]), 'Time:', datetime.datetime.now() - start_time)'''

predictions = pandas.DataFrame(pred_emotions)

def get_accuracy_score(y, prediction, Test_features_agreement):
    acc_score = 0.0
    ind = 0
    total = 0
    for index, row in prediction.iterrows():
        curr_max, ind_max = 0.0, ''
        for em in prediction.columns:
            if row[em] > curr_max:
                curr_max = row[em]
                ind_max = em
        if Test_features_agreement.iloc[ind] > 0.6:
            acc_score += y.iloc[ind,:][ind_max]
            total += 1
        ind += 1
    
    return acc_score / total

def get_multiclass_target(y):
    res = pandas.DataFrame()
    d = {'Anger':0, 'Sad':1, 'Disgust':2, 'Happy':3, 'Scared':4, 'Neutral':5}
    for em in y:
        res[em] = y[em] * d[em]
    return res.sum(axis=1)

def prediction_postprocessing(prediction):
    for index, row in prediction.iterrows():
        curr_max, ind_max = 0.0, ''
        for em in prediction.columns:
            if row[em] > curr_max:
                curr_max = row[em]
                ind_max = em
        for em in prediction.columns:
            if em == ind_max:
                row[em] = 1
            else:
                row[em] = 0

def get_prediction(ids, clf):
    for file_id in ids:
        Test_features = make_features_df_for_ids([file_id])
        
        Y_test = Test_features[columns_objects]
        Test_features_agreement = Test_features.loc[:, 'Agreement score']
        Test_features = Test_features[columns_features].drop(['Agreement score'], axis = 1)
        
        X_test = np.array(Test_features)
        X_test_scaled = scaler.fit_transform(X_test)
        pred_emotions = clf.predict_proba(X_test_scaled)
        predictions = pandas.DataFrame(pred_emotions)
        predictions = predictions.rename(columns=names)
        prediction_postprocessing(predictions)
        print(predictions.sum())
        print(file_id, get_accuracy_score(Y_test, predictions, Test_features_agreement))
    

Y_multiclass_train = get_multiclass_target(Y_train)
names = {0:'Anger', 1:'Sad', 2:'Disgust', 3:'Happy', 4:'Scared', 5:'Neutral'}
for C in [10**p for p in range(-2, -1)]:
    start_time = datetime.datetime.now()
    clf = LogisticRegression(C=C, random_state=RANDOM_SEED)
    clf.fit(X_scaled, np.array(Y_multiclass_train))
    #pred_emotions = clf.predict_proba(X_test_scaled)
    #predictions = pandas.DataFrame(pred_emotions)
    print('C=', C, 'Time:', datetime.datetime.now() - start_time)
    #predictions = predictions.rename(columns=names)
    #print(get_accuracy_score(Y_test, predictions))
    get_prediction(Test_ids, clf)
'''
x_range = range(0, len(Y_test))
for emotion in Y_test:
    plot_pred(x_range, Y_test.loc[:, emotion], pred_emotions[emotion] , "index", "result_" + emotion)'''