
# Use this file to create a submission

if __name__ == '__main__':
    import pandas as pd
    scores = pd.read_csv('myScoreFile.txt', sep='\t', names=['srch_id', 'orig_rank', 'rank_score'])
    test = pd.read_hdf('/scratch/code/elvan/DataMiningTechniques/Assignment2/data/data.h5', 'test')

    test['rank_score'] = scores.rank_score.values
    test = test.groupby('srch_id').apply(lambda g: g.sort_values('rank_score', ascending=False))

    del test['rank_score']

    test.to_csv('test_submission.csv')


