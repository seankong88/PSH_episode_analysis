import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import os
import glob
import numpy as np
import csv 
import datetime
import time

import statistics as st
import scipy.stats as stats
import math

import scipy.io as spio
from datetime import datetime, timedelta
import functools as ft

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler
from sklearn.model_selection import train_test_split

import util.psh_util as psh_util

###############################################
# all helper methods for model-based analysis #
###############################################



'''
extract feature with sliding window
FOR TESTING ON NEW DATA
'''
def extract_feature_sliding_window(df, vital_type, window_size, overlap):
    
    #data = df[vital_type]
    agg_df = []
    
    start_idx_lst = []
    end_idx_lst = []
    
    start = df.iloc[0].time_epoch
    end = df.iloc[-1].time_epoch
    
    #print(end)
    
    left = start
    right = left + window_size
    
    while left < end:
        right = min(right, end)
        cur_data = df[df['time_epoch'].between(left, right)]
        
        if not cur_data.empty:
            agg = cur_data[vital_type].aggregate(['count', 'mean', 'std', 'var', 'min', 'max', 'median'])
            agg = agg.to_frame().T
            agg_df.append(agg)
            
            start_index = cur_data['index'].iloc[0]
            end_index = cur_data['index'].iloc[-1]
            
            start_idx_lst.append(start_index)
            end_idx_lst.append(end_index)
            
        left = left + int(window_size*overlap)
        right = left + window_size
        #print('left:'+str(left)+',right:'+str(right))
        
    final_df = pd.concat(agg_df)
    
    final_df['start'] = start_idx_lst
    final_df['end'] = end_idx_lst
    
    return final_df

'''
combine indices for testing data, after each sliding window is labelled
'''
def combine_indices_for_result(test, total_len):
    labels = [0]*total_len
    
    for index, row in test.iterrows():
        start_idx = int(row['start'])
        end_idx = int(row['end'])
        
        if row['label'] == 1:
            #print('s: '+str(start_idx)+' ,e: '+str(end_idx))
            for i in range(start_idx, end_idx+1): 
                labels[i] += 1
            
    return labels

'''
flatten label list for overlapping 
'''
def flatten_labels(labels):
    for i in range(len(labels)):
        if labels[i] != 0:
            labels[i] = 1
            
    return np.array(labels)


'''
helper method to return labelled intervals on time series data
'''
def find_intervals_from_list(labels):
    intervals = []
    
    start_idx = 0
    end_idx = start_idx
    
    n = len(labels)
    
    flag = False
        
    for i in range(n):
        if labels[i] == 1:
            if flag:
                end_idx += 1
            else:
                start_idx = i
                end_idx = start_idx
                flag = True
        elif labels[i] == 0:
            if flag:
                intervals.append([start_idx, end_idx])
                flag = False                
    
    return intervals


'''
SVM model based analysis 
1. Training SVM model using ground truth features
2. Sliding window on other patients' vital, extract features
3. Label each sliding window, combine results
'''
def compute_all_patients(all_patient_meta_pd, feature_list, all_features, vital_type, window_size, overlap):
    all_pd = []
    all_burden = []
    
    '''
    Part I. training model
    '''
    X_train = all_features[feature_list]
    y_train = all_features.label
    
    logreg = LogisticRegression(random_state=1)
    logreg.fit(X_train, y_train)
    
    for i in range(len(all_patient_meta_pd)):
    # for i in range(len(patient_list)):
    #for i in range(2):
    
        pid = int(all_patient_meta_pd.iloc[i].record_id)
        if np.isnan(all_patient_meta_pd.iloc[i].PSH_status):
            psh_status = 0
        else:
            psh_status = int(all_patient_meta_pd.iloc[i].PSH_status)
        #psh_status = int(all_patient_meta_pd.iloc[i].PSH_status)
    
        print('current id: '+str(pid))
        print('PSH status: '+str(psh_status))
        
        '''
        Part II. create intervals
        '''
        # create the driver vital: SPO2r
        url = '/trend/'+str(pid)+'_trend.mat'
        cur_hr = psh_util.create_vital_from_url(url, vital_type)
        
        total_len = len(cur_hr)
        
        Y_all = extract_feature_sliding_window(cur_hr, vital_type, window_size, overlap)
        Y_all = Y_all.dropna()
        
        X_test = Y_all[feature_list]
        #X_test = X_test.dropna()
        y_pred = logreg.predict(X_test)
        y_prob = logreg.predict_proba(X_test)
        
        Y_all['label'] = y_pred
        
        y_labels = combine_indices_for_result(Y_all, total_len)
        y_labels_array = flatten_labels(y_labels)
        
        y_label_intervals = find_intervals_from_list(y_labels_array)
        
        length = len(y_label_intervals)
        
        cur_burden = sum(y_labels_array) / total_len
        print('burden_score: '+str(cur_burden))
    
        '''
        Part III. combine result
        '''
        # all current ids
        id_lst = length*[pid]
        # psh status list
        psh_lst = length*[psh_status]
    
        start_time_lst = []
        end_time_lst = []
    
        start_index_lst = []
        end_index_lst = []
        
        duration_lst = []
                
        for s, e in y_label_intervals:
            start_index_lst.append(s)
            end_index_lst.append(e)
            
            interval_df = cur_hr[cur_hr['index'].between(s, e)]
            start_time_lst.append(interval_df.iloc[0]['time_epoch'])
            end_time_lst.append(interval_df.iloc[-1]['time_epoch'])
            
            cur_duration = interval_df.iloc[-1]['time_epoch'] - interval_df.iloc[0]['time_epoch']
            
            duration_lst.append(cur_duration)
            
        cur_interval_pd = pd.DataFrame({'id': id_lst, 'PSH': psh_lst, 'start_time': start_time_lst, 
                                        'end_time': end_time_lst, 'start_index': start_index_lst, 
                                        'end_index': end_index_lst, 'duration': duration_lst})
        
        #all_pd = pd.concat([all_pd, cur_interval_pd], ignore_index=True)
        
        cur_burden_pd = pd.DataFrame({'id': pid, 'PSH': psh_status, 'burden': cur_burden}, index=[0])
        
        all_pd.append(cur_interval_pd)
        all_burden.append(cur_burden_pd)
        print('-------')
    
    final_pd = pd.concat(all_pd)
    final_burden_pd = pd.concat(all_burden)
    final_pd.to_csv('model_based_all_patients_intervals.csv', index=False)
    print('complete!')
    
    return final_pd, final_burden_pd

'''
SVM model based analysis for the first 14 days of hospital stay
'''
def compute_all_patients_14(all_patient_meta_pd, feature_list, all_features, vital_type, window_size, overlap):
    all_pd = []
    all_burden = []
    
    '''
    Part I. training model
    '''
    X_train = all_features[feature_list]
    y_train = all_features.label
    
    logreg = LogisticRegression(random_state=1)
    logreg.fit(X_train, y_train)
    
    for i in range(len(all_patient_meta_pd)):
    # for i in range(len(patient_list)):
    #for i in range(2):
    
        pid = int(all_patient_meta_pd.iloc[i].record_id)
        if np.isnan(all_patient_meta_pd.iloc[i].PSH_status):
            psh_status = 0
        else:
            psh_status = int(all_patient_meta_pd.iloc[i].PSH_status)
        #psh_status = int(all_patient_meta_pd.iloc[i].PSH_status)
    
        print('current id: '+str(pid))
        print('PSH status: '+str(psh_status))
        
        '''
        Part II. create intervals
        '''
        # create the driver vital: SPO2r
        url = '/trend/'+str(pid)+'_trend.mat'
        cur_hr = psh_util.create_vital_from_url(url, vital_type)
        
        initial_time = cur_hr.iloc[0]['time_epoch']
        end_time = 14*24*60*60 + initial_time
    
        cur_hr = cur_hr[cur_hr['time_epoch'].between(initial_time, end_time)]
        
        total_len = len(cur_hr)
        
        print(total_len)
        
        Y_all = extract_feature_sliding_window(cur_hr, vital_type, window_size, overlap)
        Y_all = Y_all.dropna()
        
        X_test = Y_all[feature_list]
        #X_test = X_test.dropna()
        y_pred = logreg.predict(X_test)
        y_prob = logreg.predict_proba(X_test)
        
        Y_all['label'] = y_pred
        
        y_labels = combine_indices_for_result(Y_all, total_len)
        y_labels_array = flatten_labels(y_labels)
        
        y_label_intervals = find_intervals_from_list(y_labels_array)
        
        length = len(y_label_intervals)
        
        cur_burden = sum(y_labels_array) / total_len
        print('burden_score: '+str(cur_burden))
    
        '''
        Part III. combine result
        '''
        # all current ids
        id_lst = length*[pid]
        # psh status list
        psh_lst = length*[psh_status]
    
        start_time_lst = []
        end_time_lst = []
    
        start_index_lst = []
        end_index_lst = []
        
        duration_lst = []
                
        for s, e in y_label_intervals:
            start_index_lst.append(s)
            end_index_lst.append(e)
            
            interval_df = cur_hr[cur_hr['index'].between(s, e)]
            start_time_lst.append(interval_df.iloc[0]['time_epoch'])
            end_time_lst.append(interval_df.iloc[-1]['time_epoch'])
            
            cur_duration = interval_df.iloc[-1]['time_epoch'] - interval_df.iloc[0]['time_epoch']
            
            duration_lst.append(cur_duration)
            
        cur_interval_pd = pd.DataFrame({'id': id_lst, 'PSH': psh_lst, 'start_time': start_time_lst, 
                                        'end_time': end_time_lst, 'start_index': start_index_lst, 
                                        'end_index': end_index_lst, 'duration': duration_lst})
        
        #all_pd = pd.concat([all_pd, cur_interval_pd], ignore_index=True)
        
        cur_burden_pd = pd.DataFrame({'id': pid, 'PSH': psh_status, 'burden': cur_burden}, index=[0])
        
        all_pd.append(cur_interval_pd)
        all_burden.append(cur_burden_pd)
        print('-------')
    
    final_pd = pd.concat(all_pd)
    final_burden_pd = pd.concat(all_burden)
    final_pd.to_csv('model_based_all_patients_intervals.csv', index=False)
    print('complete!')
    
    return final_pd, final_burden_pd

