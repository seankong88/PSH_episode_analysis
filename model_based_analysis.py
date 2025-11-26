import util.psh_util as psh_util
import model_based_detector as mb
import pandas as pd

def model_based_main():
    # get raw vital HR (spo2r) from mat file
    url1 = '/trend/1_trend.mat'
    spo2r1 = psh_util.create_vital_from_url(url1, 'SPO2r')

    # get annotations (ground truth), label only from Auton viewer
    ground_truth = pd.read_csv('/data/all_annotations_GT.csv')
    p1_annotations = ground_truth[ground_truth.project_id == 9]

    # all annotations, magic number = 1675827557.07 for converting timestamp from viewer to proper epoch
    spo2r_int = psh_util.create_annotation_intervals(p1_annotations, 1675827557.07)

    # get annotations on raw trend data
    gt_all, gt_index_all = psh_util.create_time_interval_for_GT(spo2r1, spo2r_int)
    ogt_all, ogt_index_all = psh_util.create_time_interval_outside_GT(spo2r1, spo2r_int)

    # extract features for model training
    outside_features = psh_util.extract_feature_from_outside_interval(spo2r1, 'SPO2r', spo2r_int)
    inside_features = psh_util.extract_feature_from_interval(spo2r1, 'SPO2r', spo2r_int)

    # apply labels, and concatenate all features for training
    label_psh = [1]*len(inside_features)
    label_non_psh = [0]*len(outside_features)
    inside_features['label'] = label_psh
    outside_features['label'] = label_non_psh
    combined_features = [outside_features, inside_features]
    all_interval_features = pd.concat(combined_features)

    # stat features
    feature_list = ['mean', 'std', 'var', 'min', 'max', 'median']

    # getting patient ids
    patient_list = pd.read_csv('/data/PSHAMscores_deidentified.csv')
    patient_ids = patient_list['pid'].unique()

    all_patient_meta = []

    for i in patient_ids:
    
        cur_pair = [int(i), patient_list[patient_list['pid'] == i].iloc[-1]['case_control']]
        all_patient_meta.append(cur_pair)
    
    all_patient_meta_pd = pd.DataFrame(all_patient_meta, columns=['record_id', 'PSH_status'])

    # each sliding window at 40 min (2400 datapoints), with 25% overlap 
    all_patients_gt, all_patients_burden = mb.compute_all_patients(all_patient_meta_pd=all_patient_meta_pd, feature_list=feature_list, 
                                                                   all_features=all_interval_features, vital_type='SPO2r', window_size=1200, overlap=0.25)
    
    patients_gt_14, patients_burden_14 = mb.compute_all_patients_14(all_patient_meta_pd=all_patient_meta_pd, feature_list=feature_list, 
                                                                   all_features=all_interval_features, vital_type='SPO2r', window_size=1200, overlap=0.25)
    
if __name__ == "__main__":

    model_based_main()