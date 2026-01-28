#src/config.py 

"""
In this file we have two lists of all the features we selected both for 
the classification and regression tasks 
The features were selected using random variable method for Regression 
and feature importance for classification. 
For more information about the feature selection method, please check notebook 4_Feature_selection.ipynb
"""

INPUT_FEATURES_CHURN = ['contacts_count_12_mon',
 'months_inactive_12_mon',
 'total_ct_chng_q4_q1',
 'total_relationship_count',
 'total_trans_ct',
 'total_trans_amt',
 'total_revolving_bal',
 'avg_open_to_buy',
 'avg_utilization_ratio',
 'marital_status',
 'gender',
 'credit_limit',
 'income_category',
 'dependent_count',
 'total_amt_chng_q4_q1',
 'customer_age']

INPUT_FEATURES_PRICE = ['full_sq',
'mkad_km',
'public_transport_station_min_walk',
'leisure_count_3000',
'total_ct_chng_q4_q1',
'cafe_sum_1000_min_price_avg',
'max_floor',
'build_year',
'culture_objects_top_25',
'state',
'life_sq',
'office_sqm_5000',
'leisure_count_5000' ,
'sport_count_2000'  ,
'railroad_station_walk_km' ,
'detention_facility_km' ,
'trc_count_2000' ,
'cafe_count_1000_price_1000' ,
'metro_km_walk' ,
'0_17_all' ,
'cafe_sum_1500_min_price_avg']

