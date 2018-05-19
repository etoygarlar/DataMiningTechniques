import pandas as pd

# This file maps the original kaggle entries to the test dataset in order to do sanity check

# read original kaggle training set
orig_kaggle = pd.read_csv('train.csv')

# read the test dataset from the course
test_course = pd.read_hdf('/scratch/code/elvan/DataMiningTechniques/Assignment2/data/data.h5', 'test')

test_course['srch_id2'] = test_course['date_time'] + test_course['site_id'].astype('str') + test_course.visitor_location_country_id.astype('str') + test_course.srch_destination_id.astype('str') + test_course.srch_length_of_stay.astype('str')
orig_kaggle['srch_id2'] = orig_kaggle['date_time'] + orig_kaggle['site_id'].astype('str') + orig_kaggle.visitor_location_country_id.astype('str') + orig_kaggle.srch_destination_id.astype('str') + orig_kaggle.srch_length_of_stay.astype('str')

# merge the kaggle/course datasets on srch_id2
merged = test_course.merge(orig_kaggle[['srch_id', 'srch_id2']], on='srch_id2', how='left', suffixes=['', 'original'])

# find the difference between the kaggle and course srch_id's
merged['sdiff'] = (merged.srch_idoriginal - merged.srch_id)
merged['sdiff'].unique()  # this will give 332787

# create annotations, which maps the test datasets' entries to the kaggle's click_bool and booking_bool
anno = orig_kaggle[orig_kaggle.srch_id.isin(test_course.srch_id.unique() + 332787)][['srch_id', 'prop_id', 'click_bool', ' ']]
anno.index = range(len(anno))
anno.srch_id -= 332787

# save this to the same file
anno.to_hdf('/scratch/code/elvan/DataMiningTechniques/Assignment2/data/data.h5', 'test_truth')

