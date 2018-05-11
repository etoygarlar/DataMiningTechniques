
import pandas as pd
import numpy as np


def chained_assignment(f):
    """
        Decorator to supress SettingWithCopy warnings
    :param f:
    :return:
    """
    def f_(*args, **kwargs):
        pd.set_option('mode.chained_assignment', None)
        return_value = f(*args, **kwargs)
        pd.set_option('mode.chained_assignment', 'warn')
        return return_value
    return f_


@chained_assignment
def compute_relevance_grades(data):
    """
        Compute the relevance grades (5 for booked, 1 for clicked, 0 for the rest)
    :param data:
    :return:
    """
    data['relevance_grade'] = 0
    data['relevance_grade'][data.click_bool == 1] = 1
    data['relevance_grade'][data.booking_bool == 1] = 5
    return data


def clean_data(data):
    data.date_time = pd.to_datetime(data.date_time)
    # more than 90% missing for these, remove them:
    del data['visitor_hist_starrating']
    del data['visitor_hist_adr_usd']

    # these are ground-truth data, ie they don't exist in the test data
    del data['position']
    del data['gross_bookings_usd']

    for c in ['prop_review_score', 'prop_location_score2']:
        data[c] = data[c].fillna(0)
    # log likelihood of appearing in search, fill NANs with double the minimum ( means not likely at all )
    data['srch_query_affinity_score'] = data.srch_query_affinity_score.fillna(data.srch_query_affinity_score.min() * 2)
    # fill the NAN distances with the median
    data['orig_destination_distance'] = data.orig_destination_distance.fillna(data.orig_destination_distance.median())

    # summarize the competitor info
    # see some clarification here:
    # - https://www.kaggle.com/c/expedia-personalized-sort/discussion/5774
    # - https://www.kaggle.com/c/expedia-personalized-sort/discussion/5690
    comp_invs = ['comp%s_inv' % (i + 1) for i in range(8)]
    log_ratios = []
    for i in range(8):
        rate_field = 'comp%s_rate' % (i + 1)
        percentage_field = 'comp%s_rate_percent_diff' % (i + 1)
        log_ratio_field = 'comp%s_log_ratio' % (i + 1)
        log_ratios.append(log_ratio_field)

        # if data doesn't exist, assume no difference
        data[log_ratio_field] = data[percentage_field].fillna(0)
        # convert to sided ratio, then the ratio becomes 100 * (competitor - expedia) / expedia
        data[log_ratio_field] *= data[rate_field].fillna(0)
        # convert to log2(competitor / expedia)
        data[log_ratio_field] = np.log2(data[log_ratio_field] / 100 + 1)
        # clip to half/double the price, assume others are outliers
        data[log_ratio_field] = data[log_ratio_field].clip(-1, 1)

        del data[rate_field]
        del data[percentage_field]
    data['comp_min_log_ratio'] = data[log_ratios].min(axis=1)
    data['comp_max_log_ratio'] = data[log_ratios].max(axis=1)
    data['comp_mean_log_ratio'] = data[log_ratios].mean(axis=1)

    # only use the positive values for comp_invs, negative means sth different, see links above
    data['comp_inv_total'] = (data[comp_invs] * (data[comp_invs] > 0)).sum(axis=1)
    for c in comp_invs:
        del data[c]

    # scale the following columns between the min/max
    per_srch_scaled_columns = ['price_usd', 'prop_starrating', 'prop_review_score',
              'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'srch_query_affinity_score']

    data.index = data.srch_id
    for c in per_srch_scaled_columns:
        data['%s_min' % c] = data.groupby('srch_id')[c].min()
        data['%s_max' % c] = data.groupby('srch_id')[c].max()
    data.index = range(len(data))

    # TODO: According to the winning entry, it helps to have avg property features, so add these features as well
    # https://www.kaggle.com/c/expedia-personalized-sort/discussion/6228

    for c in per_srch_scaled_columns:
        data['%s_normalized' % c] = (data[c] - data['%s_min' % c]) / (data['%s_max' % c] - data['%s_min' % c])
        del data[c]

    return data


def split_data(data, train_ratio=0.8):
    """
        Split the data into train/test sets

    :param data: All the labeled data available
    :param train_ratio: ratio of training set to the whole dataset
    :return:
    """
    ids = data.srch_id.unique()
    id_cnt = len(ids)
    train_ids = np.random.choice(ids, size=int(id_cnt*train_ratio), replace=False)
    train_data = data[data.srch_id.isin(train_ids)]

    # these are ground-truth data, ie they don't exist in the actual test data,
    # we only remove from train_data since we will use the relevance_grade to evaluate
    del train_data['click_bool']
    del train_data['booking_bool']
    del train_data['relevance_grade']

    test_data = data[~data.srch_id.isin(train_ids)]
    return train_data, test_data


def compute_score(predictions):
    from metrics import ndcg
    return ndcg(predictions.relevance_grade, 38)


def evaluate(predictions):
    return predictions.groupby('srch_id').apply(compute_score).mean()


def train(train_data):
    #TODO: train with ranklib or pyltr or any other from this link: http://arogozhnikov.github.io/2015/06/26/learning-to-rank-software-datasets.html
    # This link has a lot of models in ranking setup with examples: https://github.com/ogrisel/notebooks/blob/master/Learning%20to%20Rank.ipynb
    return model


if __name__ == '__main__':
    data = pd.read_hdf('data/data.h5', 'train')
    data = compute_relevance_grades(data)

    data = clean_data(data)
    # TODO: create save_data() w.r.t https://sourceforge.net/p/lemur/wiki/RankLib%20File%20Format/

    print("Ratio of nans per feature %s" % (data.isnull().sum(axis=0) / len(data)))

    train_data, test_data = split_data(data, 0.997)
    model = train(train_data)
    test_data = model.predict(test_data)
    score = evaluate(test_data)

    print(test_data)