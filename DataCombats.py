import numpy as np
import pandas
import matplotlib.pyplot as plt
from os.path import isfile, join
from os import listdir, makedirs
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score
import DataCombatsPlot
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
    # labels or predictions
    labels_file = '{}labels/{}'.format(directory_path, file_name)
    labels = read_file_csv(labels_file)
    if labels is not None:
        result_features_list.append(labels)
    else:
         predictions_file = '{}prediction/{}'.format(directory_path, file_name)
         predictions = read_file_csv(predictions_file)
         if predictions is not None:
             result_features_list.append(predictions)
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
        eyes = pandas.read_csv('{}eyes/{}'.format(directory_path, file_name), skiprows=[0], header=None)
        eyes = rename_columns(eyes, 'Time', 'eyes')
        return eyes
    elif (data_type == 'face_nn'):
        face_nn = pandas.read_csv('{}face_nn/{}'.format(directory_path, file_name), skiprows=[0], header=None)
        face_nn = rename_columns(face_nn, 'Time', 'face')
        return face_nn
    else:
        return pandas.read_csv('{}{}/{}'.format(directory_path, data_type, file_name))

def make_id_from_filename(file_name):
    """
        Convert file name to number. Assuming that file name has format: 'id*.csv'
    """
    return int(file_name[2:-4], 16)

def make_features_df_for_ids(ids, directory_path, template_df = None, agreement_score=None, verbose=False, create_id_column=False):
    """
        Makes dataframe from list of @ids using only rows, where Agreement score >= @agreement_score
    """
    if verbose:
        print('Start make_features_df_for_ids. Total files to process:', len(ids), 'directory path: ', directory_path)
    start_time = datetime.datetime.now()
    res = pandas.DataFrame()
    ind = 0
    for file_name in ids:
        # Получаем все признаки в куче
        if verbose:
            print("make_features_df_for_ids. index: {}. dir: {}. id: {}".format(ind, directory_path, file_name))
        features = get_features_for_file_name(directory_path, file_name, template_df)
        features = features.drop(['Time'], axis=1)
        if (create_id_column):
            features['id'] = file_name
        #features = features.iloc[::7,:]
        if agreement_score:
            features = features[features['Agreement score'] >= agreement_score]
        if ind > 0:
            res = res.append(features)
        else:
            res = features
        ind += 1
    if verbose:
        print('Done. Total time:', datetime.datetime.now() - start_time, '\n')
    return res

def get_files_without_few_features(files_df):
    for index, row in files_df.iterrows():
        if (row[files_df.columns].sum() <= 2):
            files_df.drop(index, inplace=True)

def make_train_test_features(directory_path, data_type_names):
    # Получаем информацию о всех обрабатываемых файлах
    files_df = get_files_dataframes(directory_path, data_type_names)
    # Фильтруем файлы без признаков или с малым набором признаков
    get_files_without_few_features(files_df)
    # Формируем шаблон для всех признаков на основе перечисленных в списке по группам признаков(data_type_names)
    template_df = get_template_dataframe(directory_path, files_df, data_type_names)
    # Формируем тестовую и тренировочную выборки
    Train_files, Test_files = train_test_split(files_df, random_state=RANDOM_SEED)
    print ('Train_files split:\n', Train_files.groupby(Train_files.columns.tolist(),as_index=False).size())
    print ('Test_files split:\n', Test_files.groupby(Test_files.columns.tolist(),as_index=False).size())
    Train_ids = Train_files.index
    Test_ids = Test_files.index
    # Получаем таблицы признаков
    Train_features = make_features_df_for_ids(Train_ids, directory_path, template_df = template_df, verbose=True)
    Test_features = make_features_df_for_ids(Test_ids, directory_path, template_df = template_df, verbose=True)
    return Train_features, Test_features, template_df

def make_train_test_features_final(directory_train_path, directory_test_path, data_train_type_names, data_test_type_names):
    files_train_df = get_files_dataframes(directory_train_path, data_train_type_names)
    get_files_without_few_features(files_train_df)
    files_test_df = get_files_dataframes(directory_test_path, data_test_type_names)
    print ('Train_files split:\n', files_train_df.groupby(files_train_df.columns.tolist(),as_index=False).size())
    print ('Test_files split:\n', files_test_df.groupby(files_test_df.columns.tolist(),as_index=False).size())    
    template_train_df = get_template_dataframe(directory_train_path, files_train_df, data_train_type_names)
    template_test_df = get_template_dataframe(directory_test_path, files_test_df, data_test_type_names)
    Train_ids = files_train_df.index
    Test_ids = files_test_df.index
    Train_features = make_features_df_for_ids(Train_ids, directory_train_path, template_df = template_train_df, verbose=True)
    Test_features = make_features_df_for_ids(Test_ids, directory_test_path, template_df = template_test_df, verbose=True, create_id_column=True)
    return Train_features, Test_features

def get_accuracy_score(y, prediction):
    acc_score = 0.0
    ind = 0
    total = 0
    for index, row in prediction.iterrows():
        curr_max, ind_max = 0.0, ''
        for em in prediction.columns:
            if row[em] > curr_max:
                curr_max = row[em]
                ind_max = em
        #if Test_features_agreement.iloc[ind] > 0.6:
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

def get_multiclass_accuracy_score(y, prediction):
    ind = 0
    counter = 0
    for cl in y:
        if (prediction[ind] == cl):
            counter+=1
        ind += 1
    return counter / len(y)

def reverse_multiclass_target(y):
    res = pandas.DataFrame(np.zeros((len(y), len(columns_objects_dic))), columns=[x for x in range(len(columns_objects_dic))])
    ind = 0
    for cl in y:
        res.loc[ind, cl] = 1
        ind += 1
    return res.rename(columns=columns_objects_dic)
    

def prediction_postprocessing(prediction):
    result_predictions = pandas.DataFrame(np.zeros((len(prediction.index), len(prediction.columns))), columns=prediction.columns)  
    ind = 0
    for index, row in prediction.iterrows():
        curr_max, ind_max = 0.0, ''
        for em in prediction.columns:
            if row[em] > curr_max:
                curr_max = row[em]
                ind_max = em
        result_predictions.loc[ind, ind_max] = 1
        ind += 1
    return result_predictions
                
def get_prediction_for_ids(ids, clf, directory_path, template_df = None):
    for file_id in ids:
        Test_features = make_features_df_for_ids([file_id], directory_path, template_df = template_df)
        
        Y_test = Test_features[columns_objects]
        #Test_features_agreement = Test_features.loc[:, 'Agreement score']
        Test_features = Test_features[columns_features].drop(['Agreement score'], axis = 1)
        
        X_test = np.array(Test_features)
        X_test_scaled = scaler.fit_transform(X_test)
        pred_emotions = clf.predict_proba(X_test_scaled)
        predictions = pandas.DataFrame(pred_emotions)
        predictions = predictions.rename(columns=columns_objects_dic)
        result_predictions = prediction_postprocessing(predictions)
        print(result_predictions.sum())
        print(file_id, get_accuracy_score(Y_test, result_predictions))
'''
data_type_names= ['audio', 'eyes', 'face_nn', 'kinect', 'labels']
directory_path = 'data/train/'
Train_features, Test_features, template_df = make_train_test_features(directory_path, data_type_names)

#cv = KFold(n_splits=5, shuffle=True)
columns_objects = ['Anger', 'Sad', 'Disgust', 'Happy', 'Scared', 'Neutral']
columns_objects_dic = {0:'Anger', 1:'Sad', 2:'Disgust', 3:'Happy', 4:'Scared', 5:'Neutral'}
columns_features = [col for col in Train_features.columns if col not in columns_objects]

Y_train = Train_features[columns_objects]
Train_features = Train_features[columns_features].drop(['Agreement score'], axis = 1)

X = np.array(Train_features)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

Y_test = Test_features[columns_objects]
Test_features_agreement = Test_features.loc[:, 'Agreement score']
Test_features = Test_features[columns_features].drop(['Agreement score'], axis = 1)

X_test = np.array(Test_features)
X_test_scaled = scaler.fit_transform(X_test)
pred_emotions = {}

predictions = pandas.DataFrame(pred_emotions)

Y_multiclass_train = get_multiclass_target(Y_train).astype(int)
Y_multiclass_test = get_multiclass_target(Y_test).astype(int)

best_C = 0
#best_prediction = pandas.DataFrame
best_prediction_probability = pandas.DataFrame()
best_result_prediction = pandas.DataFrame()
best_accuracy_LR = 0    
best_accuracy_RF_C_for_LR = 0
best_criterion_for_LR = ''
best_n_estimator_for_LR = 0

#%%

criterions = ['gini', 'entropy']
n_estimators = [x*20 for x in range(1,16)]
for C in [10**p for p in range(-5, 3)]:
    print ('C =', C, "Learning process...")
    start_time = datetime.datetime.now()
    clf = LogisticRegression(C=C, random_state=RANDOM_SEED, n_jobs = -1)
    clf.fit(X_scaled, np.array(Y_multiclass_train))
    
    #pred_emotion = reverse_multiclass_target(clf.predict(X_test_scaled))
    pred_emotions_probability = clf.predict_proba(X_test_scaled)
    predictions = pandas.DataFrame(pred_emotions_probability)
    predictions = predictions.rename(columns=columns_objects_dic)
    
    print('C =', C, 'Time:', datetime.datetime.now() - start_time)
    X_train_RF, X_test_RF, y_multiclass_train_RF, y_multiclass_test_RF = train_test_split(predictions, Y_multiclass_test.astype(int), random_state=RANDOM_SEED)
    
    best_accuracy_RF_C = 0
    best_criterion = ''
    best_n_estimator = 0
    
    for criterion in criterions:
        for n_estimator in n_estimators:
            clf_RF = RandomForestClassifier(n_estimators=n_estimator, criterion=criterion, random_state=RANDOM_SEED, n_jobs = -1)
            clf_RF.fit(X_train_RF, np.array(y_multiclass_train_RF))
            predictions_RF = clf_RF.predict(X_test_RF)
            accuracy_score_RF = get_multiclass_accuracy_score(y_multiclass_test_RF, predictions_RF)
            if (accuracy_score_RF > best_accuracy_RF_C):
                best_accuracy_RF_C = accuracy_score_RF
                best_criterion = criterion
                best_n_estimator = n_estimator
    print ('Best RF. criterion', best_criterion, 'n_estimator', best_n_estimator, 'accuracy score RF = ', best_accuracy_RF_C)
        
    accuracy_score = get_accuracy_score(Y_test, predictions)
    if (accuracy_score > best_accuracy_LR):
        best_accuracy_LR = accuracy_score
        best_C = C
        best_prediction_probability = predictions
        #best_prediction = pred_emotion
        best_result_prediction = prediction_postprocessing(predictions)
        best_accuracy_RF_C_for_LR = best_accuracy_RF_C
        best_criterion_for_LR = best_criterion
        best_n_estimator_for_LR = best_n_estimator
    print('Accuracy score LR =', accuracy_score, '\n')    
    
    #get_prediction_for_ids(Test_ids, clf, directory_path, template_df = template_df)

print('Best C =', best_C, 'Accuracy score LR=', best_accuracy_LR, '\n')
print('Best RF for C. criterion', best_criterion_for_LR, 'n_estimator', best_n_estimator_for_LR, 'accuracy score RF = ', best_accuracy_RF_C_for_LR)

print("test predictions values count:")
print(Y_test.apply(pandas.value_counts), '\n')
print("result predictions from probability values count:")
print(best_result_prediction.apply(pandas.value_counts), '\n')
#print("result predictions from predict values count:")
#print(best_prediction.apply(pandas.value_counts), '\n')
#print("Check equality:",(best_prediction != best_result_prediction).any())


# Отрисовка всех графиков предсказаний эмоций
#x_range = range(0, len(Y_test))
#for emotion in Y_test:
#    DataCombatsPlot.plot_pred(x_range, Y_test.loc[:, emotion], predictions[emotion] , "Index", "Emotion: " + emotion)

# Отрисовка графика предсказания эмоции по классам
#DataCombatsPlot.plot_pred(x_range, get_multiclass_target(Y_test), get_multiclass_target(best_result_prediction) , "Index", "Emotions")

# Отрисовка roc_auc
#DataCombatsPlot.plot_roc_auc(columns_objects, best_prediction_probability, Y_test)
'''


# Получение признаков тренировочных и тестовых в куче
data_train_type_names= ['audio', 'eyes', 'face_nn', 'kinect', 'labels']
data_test_type_names= ['audio', 'eyes', 'face_nn', 'kinect', 'prediction']
directory_train_path = 'data/train/'
directory_test_path = 'data/test/'
Train_features, Test_features = make_train_test_features_final(directory_train_path, directory_test_path, 
                                                               data_train_type_names, data_test_type_names)

# Формируем списки имен столбцов признаков и целевых
columns_objects = ['Anger', 'Sad', 'Disgust', 'Happy', 'Scared', 'Neutral']
columns_objects_dic = {0:'Anger', 1:'Sad', 2:'Disgust', 3:'Happy', 4:'Scared', 5:'Neutral'}
columns_features = [col for col in Train_features.columns if col not in columns_objects]

# Подготавливаем тренировочные данные
#Y
Y_train = Train_features[columns_objects]
Y_multiclass_train = get_multiclass_target(Y_train).astype(int)
#X
Train_features = Train_features[columns_features].drop(['Agreement score'], axis = 1)
X = np.array(Train_features)
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X)

# Подготавливаем тестовые данные
id_features_column = Test_features['id']
#X
Test_features.drop(['id'], axis=1, inplace=True)
Test_features.drop(columns_objects, axis=1, inplace=True)
X_test = np.array(Test_features)
#X_test_scaled = scaler.fit_transform(X_test)


pred_emotions = {}
predictions = pandas.DataFrame(pred_emotions)

print ("Learning process...")
start_time = datetime.datetime.now()
# Обучаем
#clf = LogisticRegression(C=C, random_state=RANDOM_SEED)
#clf.fit(X_scaled, np.array(Y_multiclass_train))
clf_RF = RandomForestClassifier(n_estimators=260, criterion='gini', random_state=RANDOM_SEED, n_jobs = -1)
clf_RF.fit(X, np.array(Y_multiclass_train))
predictions_RF = clf_RF.predict(X_test)
# Предсказываем
print ("Prediction process...")
print('Time:', datetime.datetime.now() - start_time)    

print ("Save prediction to file process...")
predictions_final = reverse_multiclass_target(predictions_RF)

predictions_final["id"]=id_features_column.values

grouped_predictions = predictions_final.groupby("id")

directory_test_path_predictions = '{}prediction_{}'.format(directory_test_path, datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S'))
makedirs(directory_test_path_predictions)
print ("Predictions save directory:", directory_test_path_predictions)
for file_id, group_predictions in grouped_predictions:
    if isfile('{}prediction/{}'.format(directory_test_path, file_id)):
        current_prediction_df = pandas.read_csv('{}prediction/{}'.format(directory_test_path, file_id), dtype={'Time': str}).fillna(0)
        current_prediction_df[columns_objects] = group_predictions[columns_objects].values.astype(int)
        current_prediction_df.to_csv('{}/{}'.format(directory_test_path_predictions, file_id), index=False)

'''
data_type_names= ['audio', 'eyes', 'face_nn', 'kinect', 'labels']
directory_path = 'data/train/'
Train_features, Test_features, template_df = make_train_test_features(directory_path, data_type_names)

#cv = KFold(n_splits=5, shuffle=True)
columns_objects = ['Anger', 'Sad', 'Disgust', 'Happy', 'Scared', 'Neutral']
columns_objects_dic = {0:'Anger', 1:'Sad', 2:'Disgust', 3:'Happy', 4:'Scared', 5:'Neutral'}
columns_features = [col for col in Train_features.columns if col not in columns_objects]

Y_train = Train_features[columns_objects]
Train_features = Train_features[columns_features].drop(['Agreement score'], axis = 1)

X = np.array(Train_features)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

Y_test = Test_features[columns_objects]
Test_features_agreement = Test_features.loc[:, 'Agreement score']
Test_features = Test_features[columns_features].drop(['Agreement score'], axis = 1)

X_test = np.array(Test_features)
X_test_scaled = scaler.fit_transform(X_test)
pred_emotions = {}

predictions = pandas.DataFrame(pred_emotions)

Y_multiclass_train = get_multiclass_target(Y_train).astype(int)
Y_multiclass_test = get_multiclass_target(Y_test).astype(int)

#%%

C=0.01
print ('C =', C, "Learning process...")
start_time = datetime.datetime.now()
clf = LogisticRegression(C=C, random_state=RANDOM_SEED)
clf.fit(X_scaled, np.array(Y_multiclass_train))
    
#pred_emotion = reverse_multiclass_target(clf.predict(X_test_scaled))
pred_emotions_probability = clf.predict_proba(X_test_scaled)
predictions = pandas.DataFrame(pred_emotions_probability)
predictions = predictions.rename(columns=columns_objects_dic)
    
print('C =', C, 'Time:', datetime.datetime.now() - start_time)
#X_train_RF, X_test_RF, y_multiclass_train_RF, y_multiclass_test_RF = train_test_split(predictions, Y_multiclass_test.astype(int), random_state=RANDOM_SEED)

criterion = 'entropy'
n_estimator = 500
print ('Learning RF. criterion', criterion, 'n_estimator', n_estimator)
clf_RF = RandomForestClassifier(n_estimators=n_estimator, criterion=criterion, random_state=RANDOM_SEED, n_jobs = -1)
clf_RF.fit(predictions, np.array(Y_multiclass_test))
#predictions_RF = clf_RF.predict(X_test_RF)
#accuracy_score_RF = get_multiclass_accuracy_score(y_multiclass_test_RF, predictions_RF)


# Получаем тестовые финальные данные из файлов
directory_test_path = 'data/test/'
data_test_type_names= ['audio', 'eyes', 'face_nn', 'kinect', 'prediction']
files_test_df = get_files_dataframes(directory_test_path, data_test_type_names)
template_test_df = get_template_dataframe(directory_test_path, files_test_df, data_test_type_names)
Test_ids = files_test_df.index
Test_features_final = make_features_df_for_ids(Test_ids, directory_test_path, template_df = template_test_df, verbose=True, create_id_column=True)

# Подготавливаем тестовые данные
id_features_column = Test_features_final['id']
#X
Test_features_final.drop(['id'], axis=1, inplace=True)
Test_features_final.drop(columns_objects, axis=1, inplace=True)
X_test_final = np.array(Test_features_final)
X_test_scaled_final = scaler.fit_transform(X_test_final)

# Предсказываем
print ("Prediction final process...")
pred_emotions_final = clf.predict_proba(X_test_scaled_final)
predictions_final = pandas.DataFrame(pred_emotions_final)
predictions_final = predictions_final.rename(columns=columns_objects_dic)
predictions_RF_final = clf_RF.predict(predictions_final)

predictions_total_final = reverse_multiclass_target(predictions_RF_final)

predictions_total_final["id"]=id_features_column.values

grouped_predictions = predictions_total_final.groupby("id")

directory_test_path_predictions = '{}prediction_{}'.format(directory_test_path, datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S'))
makedirs(directory_test_path_predictions)
print ("Predictions save directory:", directory_test_path_predictions)
for file_id, group_predictions in grouped_predictions:
    if isfile('{}prediction/{}'.format(directory_test_path, file_id)):
        current_prediction_df = pandas.read_csv('{}prediction/{}'.format(directory_test_path, file_id), dtype={'Time': str}).fillna(0)
        current_prediction_df[columns_objects] = group_predictions[columns_objects].values.astype(int)
        current_prediction_df.to_csv('{}/{}'.format(directory_test_path_predictions, file_id), index=False)
'''
'''

data_type_names= ['audio', 'eyes', 'face_nn', 'kinect', 'labels']
directory_path = 'data/train/'
Train_features, Test_features, template_df = make_train_test_features(directory_path, data_type_names)

#cv = KFold(n_splits=5, shuffle=True)
columns_objects = ['Anger', 'Sad', 'Disgust', 'Happy', 'Scared', 'Neutral']
columns_objects_dic = {0:'Anger', 1:'Sad', 2:'Disgust', 3:'Happy', 4:'Scared', 5:'Neutral'}
columns_features = [col for col in Train_features.columns if col not in columns_objects]

Y_train = Train_features[columns_objects]
Train_features = Train_features[columns_features].drop(['Agreement score'], axis = 1)

X = np.array(Train_features)
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X)

Y_test = Test_features[columns_objects]
Test_features_agreement = Test_features.loc[:, 'Agreement score']
Test_features = Test_features[columns_features].drop(['Agreement score'], axis = 1)

X_test = np.array(Test_features)
#X_test_scaled = scaler.fit_transform(X_test)
pred_emotions = {}

Y_multiclass_train = get_multiclass_target(Y_train).astype(int)
Y_multiclass_test = get_multiclass_target(Y_test).astype(int)

best_accuracy_RF = 0
best_criterion_for_LR = ''
best_n_estimator_for_LR = 0

#%%

criterions = ['gini', 'entropy']
n_estimators = [x*20 for x in range(1,16)]
best_accuracy_RF = 0
best_criterion = ''
best_n_estimator = 0
    
for criterion in criterions:
    for n_estimator in n_estimators:
        print ('RF. criterion', criterion, 'n_estimator', n_estimator)
        start_time = datetime.datetime.now()
   
        clf_RF = RandomForestClassifier(n_estimators=n_estimator, criterion=criterion, random_state=RANDOM_SEED, n_jobs = -1)
        clf_RF.fit(X, np.array(Y_multiclass_train))
        predictions_RF = clf_RF.predict(X_test)
        accuracy_score_RF = get_multiclass_accuracy_score(Y_multiclass_test, predictions_RF)
        if (accuracy_score_RF > best_accuracy_RF):
            best_accuracy_RF = accuracy_score_RF
            best_criterion = criterion
            best_n_estimator = n_estimator
        print('RF. Time:', datetime.datetime.now() - start_time, 'accuracy_score:', accuracy_score_RF)
print ('Best RF. criterion', best_criterion, 'n_estimator', best_n_estimator, 'accuracy score RF = ', best_accuracy_RF)
    
    #get_prediction_for_ids(Test_ids, clf, directory_path, template_df = template_df)

print("test predictions values count:")
print(Y_test.apply(pandas.value_counts), '\n')
print("result predictions from probability values count:")
print(best_result_prediction.apply(pandas.value_counts), '\n')

#print("result predictions from predict values count:")
#print(best_prediction.apply(pandas.value_counts), '\n')
#print("Check equality:",(best_prediction != best_result_prediction).any())


# Отрисовка всех графиков предсказаний эмоций
#x_range = range(0, len(Y_test))
#for emotion in Y_test:
#    DataCombatsPlot.plot_pred(x_range, Y_test.loc[:, emotion], predictions[emotion] , "Index", "Emotion: " + emotion)

# Отрисовка графика предсказания эмоции по классам
#DataCombatsPlot.plot_pred(x_range, get_multiclass_target(Y_test), get_multiclass_target(best_result_prediction) , "Index", "Emotions")

# Отрисовка roc_auc
#DataCombatsPlot.plot_roc_auc(columns_objects, best_prediction_probability, Y_test)
'''