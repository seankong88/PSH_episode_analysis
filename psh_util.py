import pandas as pd
import scipy.io as spio
from datetime import datetime, timedelta
import numpy as np

'''
Util method to clean up time stamp
'''
def process_time(time):
    N = len(time)
    
    for i in range(N):
        if time[i].microsecond > 5*10^5:
            #print('idx:'+str(i))
            nxt_sec = time[i].second+1
            #print(nxt_sec)
            if nxt_sec == 60:
                updated_time = datetime(time[i].year, time[i].month, time[i].day, time[i].hour, time[i].minute+1, 0)
            else:
                updated_time = datetime(time[i].year, time[i].month, time[i].day, time[i].hour, time[i].minute, nxt_sec)
            time[i] = updated_time
        elif time[i].microsecond > 0:
            # round off the microsecond
            updated_time = datetime(time[i].year, time[i].month, time[i].day, time[i].hour, time[i].minute, time[i].second)
            time[i] = updated_time

'''
Create trend data dataframe, with both timestamp and epoch time
'''
def create_vital_pd(trend, vital):
    vital_time = vital+'_time'
    
    time_arr = trend[vital_time].values[0].flatten()
    data_arr = trend[vital].values[0].flatten()
    
    vital_time_lst = []
    epoch_new_lst = []

    for i in range(len(time_arr)):
        cur_epoch = time_arr[i]*86400-62167305600
        cur_time = datetime.fromtimestamp(cur_epoch)
    
        vital_time_lst.append(cur_time)
        epoch_new_lst.append(cur_epoch)
        
    process_time(vital_time_lst)

    vital_pd = pd.DataFrame({'time':vital_time_lst, 'time_epoch':epoch_new_lst, vital:data_arr})
    
    return vital_pd

'''
Load data from url
'''
def create_vital_from_url(url, vital_type):
    raw_trend = spio.loadmat(url)
    trend = pd.DataFrame(np.hstack(raw_trend['data']))

    new_vital = create_vital_pd(trend, vital_type)
    new_vital.loc[:,'index'] = new_vital.index
    
    return new_vital


'''
Create epoch time intervals for annotations (Ground Truth)
magic number: value to convert hospital data into epoch
'''
def create_annotation_intervals(anno_df, magic_number):    
    intervals = []
    
    for i, row in anno_df.iterrows():
        start = row['left']-magic_number
        end = row['right']-magic_number
        
        intervals.append([start, end])
        
    intervals.sort(key = lambda l:l[0])
    return intervals

'''
create dataset for values within each annotation (Ground Truth)
'''
def create_time_interval_for_GT(df, intervals):
    data_arr = []
    index_arr = []
    
    for left, right in intervals:
        cur = df[df['time_epoch'].between(left, right)]
        data_arr.append(list(cur['SPO2r']))
        index_arr.append([cur['index'].iloc[0], cur['index'].iloc[-1]])
        
    return data_arr, index_arr

'''
create dataset for values outside each annotation (Ground Truth)
'''
def create_time_interval_outside_GT(df, intervals):
    data_arr = []
    index_arr = []
    
    N = len(df)
    
    start, end = 0, N
    
    for left, right in intervals:
        cur = df[df['time_epoch'].between(start, left)]
        if len(cur) != 0:
            data_arr.append(list(cur['SPO2r']))
            index_arr.append([cur['index'].iloc[0], cur['index'].iloc[-1]])
        start = right
        
    last_df = df[df['time_epoch'].between(start, end-1)]
    if len(last_df) != 0:
        data_arr.append(list(last_df['SPO2r']))
        index_arr.append([cur['index'].iloc[0], cur['index'].iloc[-1]])
    
    return data_arr, index_arr

'''
extract statistical features from each groud truth annotation
FOR MODEL TRAINING
'''
def extract_feature_from_interval(df, vital_type, intervals):
    
    #data = df[vital_type]
    agg_df = []
    
    start_idx_lst = []
    end_idx_lst = []
    
    for s, e in intervals:
        cur_data = df[df['time_epoch'].between(s, e)]
        
        if not cur_data.empty:
            agg = cur_data[vital_type].aggregate(['count', 'mean', 'std', 'var', 'min', 'max', 'median'])
            agg = agg.to_frame().T
            agg_df.append(agg)
            
            start_index = cur_data['index'].iloc[0]
            end_index = cur_data['index'].iloc[-1]
            
            start_idx_lst.append(start_index)
            end_idx_lst.append(end_index)
            
    final_df = pd.concat(agg_df)
    
    final_df['start'] = start_idx_lst
    final_df['end'] = end_idx_lst
    
    return final_df


'''
extract statistical features from each groud truth annotation
FOR MODEL TRAINING
'''
def extract_feature_from_outside_interval(df, vital_type, intervals):
    
    #data = df[vital_type]
    agg_df = []
    
    start_idx_lst = []
    end_idx_lst = []
    
    start = df.iloc[0].time_epoch
    end = df.iloc[-1].time_epoch
    
    for left, right in intervals:
        cur_data = df[df['time_epoch'].between(start, left)]
        
        if not cur_data.empty:
            agg = cur_data[vital_type].aggregate(['count', 'mean', 'std', 'var', 'min', 'max', 'median'])
            agg = agg.to_frame().T
            agg_df.append(agg)
            
            start_index = cur_data['index'].iloc[0]
            end_index = cur_data['index'].iloc[-1]
            
            start_idx_lst.append(start_index)
            end_idx_lst.append(end_index)
            
        start = right
        
    last_data = df[df['time_epoch'].between(start, end)]
    last_agg = last_data[vital_type].aggregate(['count', 'mean', 'std', 'var', 'min', 'max', 'median'])
    last_agg = last_agg.to_frame().T
    agg_df.append(last_agg)
    
    start_index = last_data['index'].iloc[0]
    end_index = last_data['index'].iloc[-1]
    
    start_idx_lst.append(start_index)
    end_idx_lst.append(end_index)
        
    final_df = pd.concat(agg_df)
    
    final_df['start'] = start_idx_lst
    final_df['end'] = end_idx_lst
    
    return final_df

