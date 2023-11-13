#!/usr/bin/python3
import json
import math
import random
import time
from random import randint

import pandas as pd
import sklearn
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from numpy import mean
from scipy.stats import sem
import seaborn as sns
from sklearn.model_selection import cross_validate, cross_val_predict, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import warnings

warnings.filterwarnings("ignore")

algorithms_names = [
    'KNN',
    'DecisionTree',
    'RandomForest',
    'Linear SVM',
    'RBF SVM',
    'MLP'
]

algorithms = [
    KNeighborsRegressor(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    SVR(kernel="linear", max_iter=10000),
    SVR(max_iter=10000),
    MLPRegressor(max_iter=10000)
]


def get_dataset(path_filename):
    df = pd.read_csv(path_filename)
    features = [
        'Car_Id', 'Person_Id', 'Trip', 'GPS_Time', 'Device_Time', 'GPS_Long',
        'GPS_Lat', 'GPS_Speed_Ms', 'GPS_HDOP', 'GPS_Bearing', 'Gx', 'Gy', 'Gz',
        'G_Calibrated', 'OBD_KPL_Average', 'OBD_Trip_KPL_Average',
        'OBD_Intake_Air_Temp_C', 'Device_Barometer_M', 'GPS_Altitude_M',
        'OBD_Engine_Load', 'OBD_Fuel_Level', 'GPS_Accuracy_M', 'OBD_Speed_Km',
        'GPS_Speed_Km', 'Device_Trip_Dist_Km', 'OBD_Engine_Coolant_Temp_C',
        'OBD_Engine_RPM', 'OBD_Adapter_Voltage', 'OBD_KPL_Instant',
        'OBD_Fuel_Flow_CCmin', 'Device_Fuel_Remaining',
        'OBD_Ambient_Air_Temp_C', 'OBD_CO2_gkm_Average', 'OBD_CO2_gkm_Instant',
        'Device_Cost_Km_Inst', 'Device_Cost_Km_Trip', 'OBD_Air_Pedal',
        'Context', 'Acceleration_kmhs', 'Reaction_Time', 'Air_Drag_Force',
        'Speed_RPM_Relation', 'KPL_Instant'
    ]
    obd_features = [
        'OBD_KPL_Average', 'OBD_Trip_KPL_Average', 'OBD_Intake_Air_Temp_C', 'OBD_Engine_Load',
        'OBD_Fuel_Level', 'OBD_Speed_Km', 'OBD_Engine_Coolant_Temp_C', 'OBD_Engine_RPM',
        'OBD_Adapter_Voltage', 'OBD_KPL_Instant', 'OBD_Fuel_Flow_CCmin', 'OBD_Ambient_Air_Temp_C',
        'OBD_CO2_gkm_Instant', 'OBD_Air_Pedal'
    ]
    # obd_features = [
    #     'OBD_Intake_Air_Temp_C', 'OBD_Engine_Load',
    #     'OBD_Fuel_Level', 'OBD_Speed_Km', 'OBD_Engine_Coolant_Temp_C', 'OBD_Engine_RPM',
    #     'OBD_Adapter_Voltage', 'OBD_CO2_gkm_Instant', 'OBD_Air_Pedal'
    # ]
    obd_features_inf = [
        'OBD_KPL_Average', 'OBD_Trip_KPL_Average', 'OBD_Intake_Air_Temp_C', 'OBD_Engine_Load',
        'OBD_Fuel_Level', 'OBD_Speed_Km', 'OBD_Engine_Coolant_Temp_C', 'OBD_Engine_RPM',
        'OBD_Adapter_Voltage', 'OBD_KPL_Instant', 'OBD_Fuel_Flow_CCmin', 'OBD_Ambient_Air_Temp_C',
        'OBD_Air_Pedal',

        'OBD_KPL_Average_entropy', 'OBD_KPL_Average_complexity', 'OBD_Trip_KPL_Average_entropy',
        'OBD_Trip_KPL_Average_complexity', 'OBD_Intake_Air_Temp_C_entropy',
        'OBD_Intake_Air_Temp_C_complexity', 'OBD_Engine_Load_entropy', 'OBD_Engine_Load_complexity',
        'OBD_Fuel_Level_entropy', 'OBD_Fuel_Level_complexity', 'OBD_Speed_Km_entropy',
        'OBD_Speed_Km_complexity', 'OBD_Engine_Coolant_Temp_C_entropy',
        'OBD_Engine_Coolant_Temp_C_complexity', 'OBD_Engine_RPM_entropy', 'OBD_Engine_RPM_complexity',
        'OBD_Adapter_Voltage_entropy', 'OBD_Adapter_Voltage_complexity', 'OBD_KPL_Instant_entropy',
        'OBD_KPL_Instant_complexity', 'OBD_Fuel_Flow_CCmin_entropy', 'OBD_Fuel_Flow_CCmin_complexity',
        # 'OBD_Ambient_Air_Temp_C_entropy', 'OBD_Ambient_Air_Temp_C_complexity',
        'OBD_CO2_gkm_Instant', 'OBD_Air_Pedal_entropy',
        'OBD_Air_Pedal_complexity'
    ]
    return df[obd_features], df[obd_features_inf].dropna()


def analyse_correlation(df, correlate_threshold):
    print('\n  # ANALYSE CORRELATION')
    corr = df.corr()
    # write into file the min and max values correlations for each feature
    with open('correlation.txt', 'w') as c:
        for feature in corr:
            c.write(f';\n{feature}\n')
            sort_corr = corr[feature].dropna().sort_values()
            try:
                c.write(f'  {sort_corr[0]}\n')
                c.write(f'  {sort_corr[-2]}\n')
            except Exception as e:
                pass
    # read min and max value for each features to array to min: (mini) and to max (maxi). To print range
    with open('correlation.txt', 'r') as c:
        results = c.read().split(';')[1:]
        mini, maxi = [], []
        for r in results:
            try:
                feature, minor, major = r.split()
                mini.append(float(minor))
                maxi.append(float(major))
            except Exception as e:
                pass
    # print('Correlation')
    # print(f'  Negative range: [{min(mini)}, {max(mini)}]')
    # print(f'  Positive range: [{min(maxi)}, {max(maxi)}]')
    # print()
    # Filter by correlation
    included = []
    excluded = []
    columns = list(df.columns)
    for i in range(len(columns)):
        c1 = df[columns[i]]
        must_add = True
        for j in range(i + 1, len(columns), 1):
            c2 = df[columns[j]]
            if abs(c1.corr(c2)) > correlate_threshold:
                must_add = False
                break
        if must_add:
            included.append(columns[i])
        else:
            excluded.append(columns[i])
    # print('correlation =', correlate_threshold, '\n')
    # print('included =', included, '#', len(included), '\n')
    # print('excluded =', excluded, '#', len(excluded), '\n')
    return corr


def plot_correlation(corr):
    print('\n  # CORRELATION')
    f, ax = plt.subplots(figsize=(10, 8))
    ax.tick_params(axis='both', which='major', labelsize=20)
    sns.set(font_scale=1.4)
    sns.heatmap(corr,
                # cmap=sns.diverging_palette(220, 10, as_cmap=True),
                vmin=-1.0, vmax=1.0,
                square=True, ax=ax)  # , cmap='crest')
    f.subplots_adjust(bottom=0.4)
    plt.matshow(corr, vmin=-1.0, vmax=1.0)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.tick_params(labelsize=14)


def plot_experiment(x, y, fig_name, y_err=None, ylim=(0.5, 1.05), path='.'):
    if y_err is None:
        y_err = [0] * len(y)
    font_size = 24
    bottom_size = 0.3
    X_axis = np.arange(len(x))

    plt.figure(fig_name, figsize=(17, 9))
    plt.bar(X_axis, y, 0.4, yerr=y_err)
    plt.ylim(ylim)
    plt.ylabel('Score', fontsize=font_size)
    plt.xticks(X_axis, x)
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -bottom_size),
    #            fancybox=True, shadow=True, ncol=3, fontsize=font_size)
    plt.subplots_adjust(bottom=bottom_size)
    plt.tick_params(axis='both', which='major', labelsize=font_size)
    plt.tick_params(axis='x', labelrotation=45)
    if path is not None:
        plt.savefig(f'{path}/{fig_name}.png')


def experiment(path):
    df_lit, df_inf = get_dataset(path)
    print(df_lit.shape, df_inf.shape)
    results_lit, results_inf = {}, {}
    target = 'OBD_CO2_gkm_Instant'
    for df, results in zip((df_lit, df_inf), (results_lit, results_inf)):
        for pred_name, predictor in zip(algorithms_names, algorithms):
            print(predictor)
            X, y = df.drop([target], axis=1), df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y)
            t0 = time.time()
            predictor.fit(X_train, y_train)
            print('time train:', time.time() - t0)
            t0 = time.time()
            # y_pred = regr.predict(X_test)
            score = predictor.score(X_test, y_test)
            time_score = time.time() - t0
            print(score)
            print('time score:', time_score)
            print()
            results[pred_name] = score
    print(len(results_lit), len(results_inf))
    return results_lit, results_inf


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # path_filename = 'VehicularData(anonymized).csv'
    file = '2.3.1'
    path_filename = f'df-folder/{file}.csv'

    # df = get_dataset(path_filename)
    # corr = analyse_correlation(df, .95)
    # plot_correlation(corr)
    # plt.show()

    # _, df = get_dataset(path_filename)
    # print(df.isna().sum())
    # print(df.shape)

    res_lit, res_inf = experiment(path_filename)
    x_lit, y_lit = [], []
    x_inf, y_inf = [], []
    for x, y, res in zip((x_lit, x_inf), (y_lit, y_inf), (res_lit, res_inf)):
        for alg_name in algorithms_names:
            x.append(alg_name)
            y.append(res[alg_name])
    plot_experiment(x_lit, y_lit, fig_name=f'score_lit__{file}', ylim=(0, 1.05))
    plot_experiment(x_inf, y_inf, fig_name=f'score_inf__{file}', ylim=(0, 1.05))
    # plt.show()
