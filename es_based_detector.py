import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import os
import glob
import numpy as np
import csv 
import time

import statistics as st
from scipy import stats
import math

import scipy.io as spio
from datetime import datetime, timedelta
import functools as ft

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold

import util.psh_util as psh_util
import util.precision_recall_ts_util as pr

'''
apply fix threshold for rolling windows
'''
def rolling_labels(df, vital_type, window_size, threshold, feature):
    
    labels = []
    data = df[vital_type]
    
    if feature == 'min':
        rolling_arr = data.rolling(window_size).min()
    elif feature == 'std':
        rolling_arr = data.rolling(window_size).std()
    elif feature == 'median':
        rolling_arr = data.rolling(window_size).median()
    elif feature == 'var':
        rolling_arr = data.rolling(window_size).var()
    
    if rolling_arr[window_size-1] > threshold:
        first_labels = [1]*window_size
        labels.extend(first_labels)
    else:
        first_labels = [0]*window_size
        labels.extend(first_labels)
        
    for i in range(window_size, len(df)):
        if rolling_arr[i] > threshold:
            labels.append(1)
        else:
            labels.append(0)
    
    # create intervals based on value
    intervals = []
    
    prev_value = labels[0]
    left_idx = -1
    
    if prev_value == 1:
        left_idx = 0
    
    for i in range(1, len(labels)):
        cur_value = labels[i]
        
        if cur_value == 1:
            if prev_value == 0:
                left_idx = i
            else:
                continue
        else:
            if prev_value == 1:
                intervals.append([left_idx, i-1])
            else:
                continue
                
        prev_value = cur_value
        
    return intervals, labels


'''
grid search for best threshold
'''
def grid_search_rules(hr, min_rule, median_rule, var_rule, meeting_index, steps, window_length, ranges, gt_labels):
    precision_lst = []
    recall_lst = []
    f1_lst = []
    
    min_lst = []
    median_lst = []
    var_lst = []
    
    min_ranges = [int(min_rule-ranges[0]), int(min_rule+ranges[1])]
    median_ranges = [int(median_rule-ranges[0]), int(median_rule+ranges[1])]
    var_ranges = [int(var_rule-ranges[0]), int(var_rule+ranges[1])]
    
    for i in range(min_ranges[0], min_ranges[1], steps):
        for j in range(median_ranges[0], median_ranges[1], steps):
            for k in range(var_ranges[0], var_ranges[1], steps):
                print('cur min: '+str(i))
                print('cur med: '+str(j))
                print('cur var: '+str(k))
                
                min_lst.append(i)
                median_lst.append(j)
                var_lst.append(k)
                
                min_interval, min_label = rolling_labels(hr, 'hr', window_length, i, 'min')
                var_interval, var_label = rolling_labels(hr, 'hr', window_length, k, 'var')
                median_interval, median_label = rolling_labels(hr, 'hr', window_length, j, 'median')
                
                combined_label = []
                
                for idx in range(len(hr)):
                    cur_l = min_label[idx]+var_label[idx]+median_label[idx]
                    if meeting_index == 1:
                        if cur_l >= 1:
                            combined_label.append(1)
                        else:
                            combined_label.append(0)
                    elif meeting_index == 2:
                        if cur_l == 2 or cur_l == 3:
                            combined_label.append(1)
                        else:
                            combined_label.append(0)
                    elif meeting_index == 3:
                        if cur_l == 3:
                            combined_label.append(1)
                        else:
                            combined_label.append(0)
                            
                gt_labels_array = np.array(gt_labels)
                combined_labels_array = np.array(combined_label)
                
                flat_metric = pr.TSMetric(metric_option="time-series", alpha_r=0.0, cardinality="reciprocal", bias_p="flat", bias_r="flat")
                precision_flat, recall_flat, f1_flat = flat_metric.score(gt_labels_array, combined_labels_array)
                print("flat metric")
                print("precision: ", precision_flat, "recall: ", recall_flat, "f1: ", f1_flat)
                print('-------')
                
                precision_lst.append(precision_flat)
                recall_lst.append(recall_flat)
                f1_lst.append(f1_flat)
                
    return pd.DataFrame({'min': min_lst, 'median': median_lst, 'var': var_lst, 'precision': precision_lst, 'recall': recall_lst, 'f1': f1_lst})

'''
merge all labelled intervals
'''
def merge(intervals):
    result = []
    for interval in sorted(intervals):
        if not result or result[-1][1] < interval[0]:
            result.append(interval)
        else:
            result[-1][1] = max(result[-1][1], interval[1])
    return result

def convert_index_time(signal, label):
    time_intervals = []
    
    for l in label:
        start_time = signal.loc[signal['index'] == l[0]].time_epoch.values[0]
        end_time = signal.loc[signal['index'] == l[1]].time_epoch.values[0]
        
        time_intervals.append([start_time, end_time])
        
    return time_intervals

# calculate confidence range for bp and rr
# compare the median inside the interval and outside the interval for x hours before and after the episode

def calculate_confidence_bp(vital, vital_type, input_interval, hours_range):
    
    in_median_lst = []
    out_median_lst = []
    
    diff_lst = {}
    
    max_diff = -1
    
    i = 0
    
    for s, e in input_interval:
        cur_data = vital[vital['time_epoch'].between(s, e)]
        
        if not cur_data.empty:
            cur_begin_time = datetime.fromtimestamp(s)
            cur_end_time = datetime.fromtimestamp(e)
            
            before_time_frame = cur_begin_time - timedelta(hours=hours_range)
            after_time_frame = cur_end_time + timedelta(hours=hours_range)
            
            before_data = vital[vital['time'].between(before_time_frame, cur_begin_time)]
            after_data = vital[vital['time'].between(cur_end_time, after_time_frame)]
            all_before_data = []
            all_after_data = []
            
            if not before_data.empty:
                before_time_epoch = before_data.iloc[0]['time_epoch']
                ints_before = list(filter(lambda interval: interval[0] >= before_time_epoch and interval[1] <= s, input_interval))
                #print('before intervals: '+str(ints_before))
                
                
                if not ints_before or len(ints_before) != 0:
                    
                    s_before = before_time_epoch
                    for si, ei in ints_before:
                        cur_before = vital[vital['time_epoch'].between(s_before, si)]
                        all_before_data.extend(cur_before[vital_type].values)
                        s_before = ei
                    all_before_data.extend(vital[vital['time_epoch'].between(s_before, s)][vital_type].values)   
                else:
                    all_before_data.extend(vital[vital['time_epoch'].between(before_time_epoch, s)][vital_type].values)
                
            if not after_data.empty:
                after_time_epoch = after_data.iloc[-1]['time_epoch']
                ints_after = list(filter(lambda interval: interval[0] >= e and interval[1] <= after_time_epoch, input_interval))
                #print('after intervals: '+str(ints_after))
                
                
                if not ints_after or len(ints_after) != 0:
                    
                    s_after = e
                    for si, ei in ints_after:
                        cur_after = vital[vital['time_epoch'].between(s_after, si)]
                        all_after_data.extend(cur_after[vital_type].values)
                        s_after = ei
                        
                    #print('here')
                    all_after_data.extend(vital[vital['time_epoch'].between(s_after, after_time_epoch)][vital_type].values)   
                else:
                    all_after_data.extend(vital[vital['time_epoch'].between(e, after_time_epoch)][vital_type].values)
            #print(all_after_data)
                
            #combined_vital = np.concatenate((before_data[vital_type].values, after_data[vital_type].values), axis=0)
            #combined_vital = combined_vital[combined_vital != 0]
               
            combined_vital = np.concatenate((all_before_data, all_after_data), axis=0)
            combined_vital = combined_vital[combined_vital != 0]
            
            outside_median = np.median(combined_vital)
            inside_median = np.median(cur_data[vital_type])
                        
            max_diff = max(max_diff, inside_median-outside_median)
            
            out_median_lst.append(outside_median)
            in_median_lst.append(inside_median)
            
            diff_lst[i] = abs(inside_median-outside_median)
        
        i += 1
                
    return diff_lst

def calculate_confidence_tmp(vital, vital_type, input_interval, threshold):
    
    result = {}
    
    i = 0
    for s, e in input_interval:
        cur_data = vital[vital['time_epoch'].between(s, e)]
        
        if not cur_data.empty:
            tmps = cur_data[vital_type].values
                        
            len_tmps = len(tmps)
            len_above_threshold = len([i for i in tmps if i >= threshold])
            
            percentage = len_above_threshold/len_tmps
            
            result[i] = percentage
            
        i += 1
        
    return result

def get_score_from_list(diff):
    
    all_values = list(diff.values())
    
    m = np.mean(all_values)
    s = np.std(all_values)

    res = {}

    for i, v in diff.items():
    
        if abs(v-m) < s:
            res[i] = 0
        elif m > v:
            res[i] = 0
        else:
            res[i] = 1
            
    return res


'''
grid search for best threshold
'''
def grid_search_rules(hr, min_rule, median_rule, var_rule, meeting_index, steps, window_length, ranges, gt_labels):
    precision_lst = []
    recall_lst = []
    f1_lst = []
    
    min_lst = []
    median_lst = []
    var_lst = []
    
    min_ranges = [int(min_rule-ranges[0]), int(min_rule+ranges[1])]
    median_ranges = [int(median_rule-ranges[0]), int(median_rule+ranges[1])]
    var_ranges = [int(var_rule-ranges[0]), int(var_rule+ranges[1])]
    
    for i in range(min_ranges[0], min_ranges[1], steps):
        for j in range(median_ranges[0], median_ranges[1], steps):
            for k in range(var_ranges[0], var_ranges[1], steps):
                print('cur min: '+str(i))
                print('cur med: '+str(j))
                print('cur var: '+str(k))
                
                min_lst.append(i)
                median_lst.append(j)
                var_lst.append(k)
                
                min_interval, min_label = rolling_labels(hr, 'hr', window_length, i, 'min')
                var_interval, var_label = rolling_labels(hr, 'hr', window_length, k, 'var')
                median_interval, median_label = rolling_labels(hr, 'hr', window_length, j, 'median')
                
                combined_label = []
                
                for idx in range(len(hr)):
                    cur_l = min_label[idx]+var_label[idx]+median_label[idx]
                    if meeting_index == 1:
                        if cur_l >= 1:
                            combined_label.append(1)
                        else:
                            combined_label.append(0)
                    elif meeting_index == 2:
                        if cur_l == 2 or cur_l == 3:
                            combined_label.append(1)
                        else:
                            combined_label.append(0)
                    elif meeting_index == 3:
                        if cur_l == 3:
                            combined_label.append(1)
                        else:
                            combined_label.append(0)
                            
                gt_labels_array = np.array(gt_labels)
                combined_labels_array = np.array(combined_label)
                
                flat_metric = pr.TSMetric(metric_option="time-series", alpha_r=0.0, cardinality="reciprocal", bias_p="flat", bias_r="flat")
                precision_flat, recall_flat, f1_flat = flat_metric.score(gt_labels_array, combined_labels_array)
                print("flat metric")
                print("precision: ", precision_flat, "recall: ", recall_flat, "f1: ", f1_flat)
                print('-------')
                
                precision_lst.append(precision_flat)
                recall_lst.append(recall_flat)
                f1_lst.append(f1_flat)
                
    return pd.DataFrame({'min': min_lst, 'median': median_lst, 'var': var_lst, 'precision': precision_lst, 'recall': recall_lst, 'f1': f1_lst})
                

