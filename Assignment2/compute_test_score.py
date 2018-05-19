
# Use this file to evaluate a models' score

if __name__ == '__main__':
    import pandas as pd
    scores = pd.read_csv('myScoreFile.txt', sep='\t', names=['srch_id', 'orig_rank', 'rank_score'])
    test = pd.read_hdf('/scratch/code/elvan/DataMiningTechniques/Assignment2/data/data.h5', 'test')

    # following is to do actual grading:
    test_truth = pd.read_hdf('/scratch/code/elvan/DataMiningTechniques/Assignment2/data/data.h5', 'test_truth')
    test = test.merge(test_truth, on=['srch_id', 'prop_id'], how='inner')
    from preprocess import compute_relevance_grades, evaluate
    test = compute_relevance_grades(test)

    test['rank_score'] = scores.rank_score.values
    test = test.groupby('srch_id').apply(lambda g: g.sort_values('rank_score', ascending=False))

    ndcg38 = evaluate(test)
    print('NDCG@38 score for the submission: %s' % ndcg38)



