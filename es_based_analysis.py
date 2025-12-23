import util.psh_util as psh_util
import es_based_detector as eb
import pandas as pd
import numpy as np
import sys

def es_based_process(labelled_case, ground_truth_annotations, patients_list_file, fileserver_path):
    # get raw vital HR (spo2r) from mat file
    #url1 = '/trend/1_trend.mat'
    url1 = labelled_case
    spo2r1 = psh_util.create_vital_from_url(url1, 'SPO2r')

    # get annotations (ground truth), label only from Auton viewer
    #ground_truth = pd.read_csv('/data/all_annotations_GT.csv')
    ground_truth = pd.read_csv(ground_truth_annotations)
    p1_annotations = ground_truth[ground_truth.project_id == 9]

    # all annotations, magic number = 1675827557.07 for converting timestamp from viewer to proper epoch
    spo2r_int = psh_util.create_annotation_intervals(p1_annotations, 1675827557.07)

    # get annotations on raw trend data
    gt_all, gt_index_all = psh_util.create_time_interval_for_GT(spo2r1, spo2r_int)
    ogt_all, ogt_index_all = psh_util.create_time_interval_outside_GT(spo2r1, spo2r_int)

    # extract features for model training
    outside_features = psh_util.extract_feature_from_outside_interval(spo2r1, 'SPO2r', spo2r_int)
    inside_features = psh_util.extract_feature_from_interval(spo2r1, 'SPO2r', spo2r_int)

    # stat features
    feature_list = ['mean', 'std', 'var', 'min', 'max', 'median']

    # getting patient ids
    patient_list = pd.read_csv(patients_list_file)
    patient_ids = patient_list['pid'].unique()

    all_patient_meta = []

    for i in patient_ids:
    
        cur_pair = [int(i), patient_list[patient_list['pid'] == i].iloc[-1]['case_control']]
        all_patient_meta.append(cur_pair)
    
    all_patient_meta_pd = pd.DataFrame(all_patient_meta, columns=['record_id', 'PSH_status'])

    # label window size: 10 min (300 datapoints)
    window_length = 300

    #############################
    ## Combine analysis result ##
    #############################
    all_pd = []

    for i in range(len(all_patient_meta_pd)):

        pid = int(all_patient_meta_pd.iloc[i].record_id)
        if np.isnan(all_patient_meta_pd.iloc[i].PSH_status):
            psh_status = 0
        else:
            psh_status = int(all_patient_meta_pd.iloc[i].PSH_status)
        #psh_status = int(all_patient_meta_pd.iloc[i].PSH_status)
    
        print('current id: '+str(pid))
        print('PSH status: '+str(psh_status))
    
        '''
            Part I. create intervals
        '''
        # create the driver vital: SPO2r
        url = fileserver_path+str(pid)+'_trend.mat'
        cur_hr = psh_util.create_vital_from_url(url, 'SPO2r')
    
        # modify min rule
        min_rule = min_rule + 10
    
        # create labelled interval on min, std, and median
        min_interval, min_label = eb.rolling_labels(cur_hr, 'SPO2r', window_length, min_rule, 'min')
        std_interval, std_label = eb.rolling_labels(cur_hr, 'SPO2r', window_length, std_rule, 'std')
        median_interval, median_label = eb.rolling_labels(cur_hr, 'SPO2r', window_length, median_rule, 'median')
    
        # combine all intervals
        print('min len: '+str(len(min_interval))+', std len: '+str(len(std_interval))+', md len: '+str(len(median_interval)))
        combined = []
        combined.extend(min_interval)
        combined.extend(std_interval)
        combined.extend(median_interval)
    
        # merge overlapping intervals
        merged = eb.merge(combined)
        print('merged len: '+str(len(merged)))
    
        # filter with short intervals, use threshold 230
        filtered = []
        for s, e in merged:
            if e-s > 230:
                filtered.append([s,e])
        print('filtered len: '+str(len(filtered)))    
    
        # convert to a time intervals
        time_intervals = eb.convert_index_time(cur_hr, filtered)
    
        # create trajectory
    
    
        '''
            Part II: get confidence scores
        '''
        # get other vitals
        # blood presure
        cur_nbps = psh_util.create_vital_from_url(url, 'NBPS')
        # RR
        cur_resp = psh_util.create_vital_from_url(url, 'RESP')
        # tmp1
        cur_tmp1 = psh_util.create_vital_from_url(url, 'TMP1')
        # tmp2
        cur_tmp2 = psh_util.create_vital_from_url(url, 'TMP2')
    
        # get confidence lists from NBPS and RESP
        # use 4 hours before/after each episode
        nbps_diff = eb.calculate_confidence_bp(cur_nbps, 'NBPS', time_intervals, 4)
        resp_diff = eb.calculate_confidence_bp(cur_resp, 'RESP', time_intervals, 4)
    
        # based on the list, calculate the actual score
        nbps_scores = eb.get_score_from_list(nbps_diff)
        resp_scores = eb.get_score_from_list(resp_diff)
    
        # get confidence lists from tmp1 and tmp2
        # use 38 degree as the threshold
        t1_percent = eb.calculate_confidence_tmp(cur_tmp1, 'TMP1', time_intervals, 38)
        t2_percent = eb.calculate_confidence_tmp(cur_tmp2, 'TMP2', time_intervals, 38)
    
        '''
            Part III: combine everything together into csv
        '''
        length = len(time_intervals)
    
        # all current ids
        id_lst = length*[pid]
        # psh status list
        psh_lst = length*[psh_status]
    
        start_time_lst = []
        end_time_lst = []
    
        start_index_lst = []
        end_index_lst = []
    
        duration = []
    
        nbps_diff_lst = []
        nbps_score_lst = []
    
        resp_diff_lst = []
        resp_score_lst = []
    
        tmp1_percent_lst = []
        tmp1_score_lst = []
    
        tmp2_percent_lst = []
        tmp2_score_lst = []
    
        for s_index, e_index in filtered:
            start_index_lst.append(s_index)
            end_index_lst.append(e_index)
    
        for idx, time_pair in enumerate(time_intervals):
            start_time_lst.append(time_pair[0])
            end_time_lst.append(time_pair[1])
        
            duration.append(time_pair[1]-time_pair[0])
        
            # adding nbps differences and score
            if idx in nbps_diff:
                nbps_diff_lst.append(nbps_diff[idx])
                nbps_score_lst.append(nbps_scores[idx])
            else:
                nbps_diff_lst.append(np.nan)
                nbps_score_lst.append(np.nan)
           
            # adding resp differences and score
            if idx in resp_diff:
                resp_diff_lst.append(resp_diff[idx])
                resp_score_lst.append(resp_scores[idx])
            else:
                resp_diff_lst.append(np.nan)
                resp_score_lst.append(np.nan)
            
            # adding tmp1 precentage and score
            if idx in t1_percent:
                tmp1_percent_lst.append(t1_percent[idx])
                if t1_percent[idx] > 0.5:
                    tmp1_score_lst.append(1)
                else:
                    tmp1_score_lst.append(0)
            else:
                tmp1_percent_lst.append(np.nan)
                tmp1_score_lst.append(np.nan)
            
            # adding tmp2 precentage and score
            if idx in t2_percent:
                tmp2_percent_lst.append(t2_percent[idx])
                if t2_percent[idx] > 0.5:
                    tmp2_score_lst.append(1)
                else:
                    tmp2_score_lst.append(0)
            else:
                tmp2_percent_lst.append(np.nan)
                tmp2_score_lst.append(np.nan)
            
        cur_interval_pd = pd.DataFrame({'id': id_lst, 'PSH': psh_lst, 'start_time': start_time_lst, 'end_time': end_time_lst, 
                                    'start_index': start_index_lst, 'end_index': end_index_lst,
                                    'duration(sec)': duration, 
                                    'NBSP_values': nbps_diff_lst, 'NBSP_scores': nbps_score_lst,
                                    'RESP_values': resp_diff_lst, 'RESP_scores': resp_score_lst,
                                    'TMP1_percent': tmp1_percent_lst, 'TMP1_score': tmp1_score_lst,
                                    'TMP2_percent': tmp2_percent_lst, 'TMP2_score': tmp2_score_lst})
    
        file_name = str(pid)+'_intervals.csv'
    
        #cur_interval_pd.to_csv(file_name, index=False)
    
        all_pd.append(cur_interval_pd)
    
        print('-----------------------')
    
    final_pd_4 = pd.concat(all_pd)
    final_pd_4.to_csv('updated_rule_all_patients_score_4hour_window.csv', index=False)
    print('complete!')

def main():
    print('Labelled Cases File Path: ', sys.argv[1])
    print('Ground Truth File Path: ', sys.argv[2])
    print('All Patients List: ', sys.argv[3])
    print('All Other Cases Path: ', sys.argv[4])
    es_based_process(labelled_case=sys.argv[1], ground_truth_annotations=sys.argv[2], patients_list_file=sys.argv[3], fileserver_path=sys.argv[4])

if __name__ == "__main__":
    main()
