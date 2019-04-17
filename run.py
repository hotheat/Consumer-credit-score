# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model import *
from feature_selection import *
from sklearn.metrics import mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor
from itertools import combinations
import xgboost as xgb
import xgbfir
import sys

pd.set_option('display.max.columns', 500)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

train_df = pd.read_csv('train_dataset/train_dataset.csv')
test_df = pd.read_csv('test_dataset/test_dataset.csv')
y_train = train_df['信用分']
x_train = train_df.drop('信用分', axis=1)
train_test = pd.concat((train_df.drop('信用分', axis=1), test_df))


def get_features(train_test):
    train_test = train_test.drop('用户编码', axis=1)
    train_test.loc[train_test['用户话费敏感度'] == 0, '用户话费敏感度'] = 5
    train_test.loc[train_test['用户年龄'] == 0, '用户年龄'] = np.nan
    train_test['年龄是否缺失'] = train_test['用户年龄'].isnull()
    train_test['是否黑名单客户'] = train_test['是否黑名单客户'].replace({0: 1, 1: 0})
    train_test['缴费用户当前是否欠费缴费'] = train_test['缴费用户当前是否欠费缴费'].replace({0: 1, 1: 0})
    train_test['当月是否去过高档商场消费'] = train_test['当月是否逛过福州仓山万达'] + train_test['当月是否到过福州山姆会员店']
    train_test.loc[train_test['当月是否去过高档商场消费'] >= 1, '当月是否去过高档商场消费'] = 1
    train_test['用户欠费缴费或黑名单'] = (
            (train_test['是否黑名单客户'] + train_test['缴费用户当前是否欠费缴费']) >= 1).apply(int)

    bi_combine_lst = ['是否经常逛商场的人', '当月是否看电影', '当月是否景点游览', '当月是否体育场馆消费']
    combine_iter = combinations(range(4), 2)

    def parse_bifeature(df, col1, col2):
        df[col1 + '或' + col2] = ((df[col1] * df[col2]) >= 1).apply(int)
        return df

    for idx1, idx2 in combine_iter:
        col1, col2 = bi_combine_lst[idx1], bi_combine_lst[idx2]
        train_test = parse_bifeature(train_test, col1, col2)

    train_test['商场_电影_景点'] = train_test['是否经常逛商场的人'] + train_test['当月是否看电影'] + train_test['当月是否景点游览']

    train_test['商场_电影_体育馆'] = train_test['是否经常逛商场的人'] + train_test['当月是否看电影'] + train_test['当月是否体育场馆消费']

    train_test['商场_体育馆_景点'] = train_test['是否经常逛商场的人'] + train_test['当月是否体育场馆消费'] + train_test['当月是否景点游览']

    train_test['体育馆_电影_景点'] = train_test['当月是否体育场馆消费'] + train_test['当月是否看电影'] + train_test['当月是否景点游览']

    train_test['体育馆_电影_景点_商场'] = train_test['当月是否体育场馆消费'] + train_test['当月是否看电影'] + train_test['当月是否景点游览'] + train_test[
        '是否经常逛商场的人']

    train_test['缴费金额是否覆盖当月账单'] = train_test['缴费用户最近一次缴费金额（元）'] - train_test['用户账单当月总费用（元）']
    train_test['最近一次缴费是否超过平均消费额'] = train_test['缴费用户最近一次缴费金额（元）'] - train_test['用户近6个月平均消费值（元）']
    train_test['当月账单是否超过平均消费额'] = train_test['用户账单当月总费用（元）'] - train_test['用户近6个月平均消费值（元）']
    train_test['交通类APP使用次数'] = train_test['当月飞机类应用使用次数'] + train_test['当月火车类应用使用次数']

    # 离散化
    lst = ['交通类APP使用次数', '当月物流快递类应用使用次数',
           '当月飞机类应用使用次数', '当月火车类应用使用次数', '当月旅游资讯类应用使用次数']

    def discreteze(x):
        if x == 0:
            return 0
        elif x <= 5:
            return 1
        elif x <= 15:
            return 2
        elif x <= 50:
            return 3
        elif x <= 100:
            return 4
        else:
            return 5

    for i in lst:
        train_test[i] = train_test[i].apply(discreteze)

    # 针对长尾分布和异常值
    def base_process(data):
        transform_value_feature = ['用户年龄', '用户网龄（月）', '当月通话交往圈人数',
                                   '最近一次缴费是否超过平均消费额', '当月账单是否超过平均消费额',
                                   '近三个月月均商场出现次数', '当月网购类应用使用次数',
                                   '当月物流快递类应用使用次数', '当月金融理财类应用使用总次数',
                                   '当月视频播放类应用使用次数', '当月飞机类应用使用次数',
                                   '当月火车类应用使用次数', '当月旅游资讯类应用使用次数',
                                   ]
        user_bill_fea = ['缴费用户最近一次缴费金额（元）', '用户近6个月平均消费值（元）',
                         '用户账单当月总费用（元）', '用户当月账户余额（元）',
                         ]
        log_features = ['当月网购类应用使用次数', '当月金融理财类应用使用总次数', '当月视频播放类应用使用次数',
                        ]
        for col in transform_value_feature + log_features + user_bill_fea:
            ulimit = np.percentile(data[col][data[col].notnull()].values, 99.9)
            llimit = np.percentile(data[col][data[col].notnull()].values, 0.1)
            data.loc[(data[col] > ulimit) & (data[col].notnull()), col] = ulimit
            data.loc[(data[col] < llimit) & (data[col].notnull()), col] = llimit

        for col in user_bill_fea + log_features:
            data[col] = np.log1p(data[col])
        return data

    train_test = base_process(train_test)
    train_test['用户网龄（年）'] = train_test['用户网龄（月）'] // 12 + 1
    agg_func = {
        '当月通话交往圈人数': ['mean'],
        '用户账单当月总费用（元）': ['mean'],
    }
    agg_time_year = train_test.groupby('用户网龄（年）').agg(agg_func)
    agg_time_year.columns = ['_'.join(col).strip()
                             for col in agg_time_year.columns.values]
    agg_time_year.reset_index(inplace=True)

    for col in agg_time_year.drop('用户网龄（年）', axis=1).columns:
        ulimit = np.percentile(agg_time_year[col].values, 99.9)
        llimit = np.percentile(agg_time_year[col].values, 0.1)
        agg_time_year.loc[agg_time_year[col] > ulimit, col] = ulimit
        agg_time_year.loc[agg_time_year[col] < llimit, col] = llimit
    train_test = pd.merge(train_test, agg_time_year, on='用户网龄（年）', how='left')

    # 账号余额 + 缴费费用
    train_test['余额+缴费费用'] = train_test['缴费用户最近一次缴费金额（元）'] + train_test['用户当月账户余额（元）']
    return train_test


def feature_importance(train_test):
    # 特征重要性 xgbfir
    x_train = train_test[:train_df.shape[0]]
    xgb_cmodel = xgb.XGBRegressor().fit(x_train.astype('float'), y_train)
    xgbfir.saveXgbFI(xgb_cmodel, feature_names=x_train.columns, OutputXlsxFile='特征重要性.xlsx')


def null_importance():
    # ## Null importance
    x_train = train_test[:train_df.shape[0]]
    ni = Null_Importance(x_train, y_train)
    actual_imp_df = ni.get_feature_importances()
    null_imp_df = ni.build_null_distribution()
    # feature = '用户网龄交往圈人数'
    # ni.display_distributions(actual_imp_df, null_imp_df, feature)
    ni.score(actual_imp_df, null_imp_df)
    ni.check_importance_unrelated_feature(actual_imp_df, null_imp_df)


def get_feature(train_test):
    train_test = get_features(train_test)
    feature_importance(train_test)
    feat_importance = pd.read_excel('特征重要性.xlsx')
    feats_selected = feat_importance['Interaction']
    train_test = train_test[feats_selected]
    return train_test


## lgb mae
lgb_params_mae = {
    'objective': 'regression_l1', 'boosting_type': 'gbdt', 'num_leaves': 42, 'reg_alpha': 3.403, 'reg_lambda': 1.559,
    'max_depth': 6, 'n_estimators': 2000, 'subsample': 0.81, 'colsample_bytree': 0.3, 'subsample_freq': 1,
    'learning_rate': 0.014, 'random_state': 2019, 'n_jobs': 8, 'min_child_weight': 37.31, 'min_child_samples': 14,
    'min_split_gain': 0.02059,
}
# lgb mse
lgb_params_mse = {
    'objective': 'regression_l2', 'boosting_type': 'gbdt', 'num_leaves': 31, 'reg_alpha': 1.2,
    'reg_lambda': 1.8, 'max_depth': 8, 'n_estimators': 2000, 'subsample': 0.8, 'colsample_bytree': 0.7,
    'subsample_freq': 1, 'learning_rate': 0.024, 'random_state': 2019, 'n_jobs': 8, 'min_child_weight': 16.4,
    'min_child_samples': 29, 'min_split_gain': 0.1,
}
# lgb 贝叶斯优化
lgb_adj_params = {
    'min_child_samples': (10, 30), 'min_split_gain': (0.001, 0.1), 'num_leaves': (24, 45),
    'feature_fraction': (0.1, 0.9), 'bagging_fraction': (0.8, 1), 'max_depth': (4, 8.99),
    'lambda_l1': (0, 5), 'lambda_l2': (0, 3), 'min_child_weight': (5, 50)
}
# xgb mae
# xgb_params_mae = {
#     'eta': 0.03, 'max_depth': 8, 'subsample': 0.75, 'colsample_bytree': 0.75,
#     'objective': 'reg:linear', 'eval_metric': 'mae', 'silent': True, 'nthread': 8,
#     'lambda': 1.5, 'alpha': 2.2, 'min_child_weight': 12, 'max_delta_step': 2,
# }
xgb_params_mae = {
    'eta': 0.09966, 'max_depth': 9, 'subsample': 0.8649, 'colsample_bytree': 0.2877, 'num_leaves': 43,
    'objective': 'reg:linear', 'eval_metric': 'mae', 'silent': True, 'nthread': 8, 'min_child_samples': 15,
    'lambda': 0.12355, 'alpha': 3.191, 'min_child_weight': 6.12, 'max_delta_step': 2, 'gamma': 2.72728,
}
# XGB MSE
# xgb_params_mse = {
#     'eta': 0.02, 'max_depth': 8, 'subsample': 0.8, 'colsample_bytree': 0.8,
#     'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True, 'nthread': 8,
#     'lambda': 1.8, 'alpha': 1.2, 'min_child_weight': 12, 'max_delta_step': 2,
# }
xgb_params_mse = {
    'eta': 0.28716, 'max_depth': 8, 'subsample': 0.51, 'colsample_bytree': 0.54773, 'num_leaves': 41,
    'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True, 'nthread': 8, 'min_child_samples': 17,
    'lambda': 1.5069, 'alpha': 3.45311, 'min_child_weight': 36.447, 'max_delta_step': 2, 'gamma': 5.96818,
}

# xgb 贝叶斯优化
xgb_adj_params = {
    'min_child_samples': (10, 30), 'num_leaves': (20, 45), 'colsample_bytree': (0.1, 1),
    'subsample': (0.5, 1), 'max_depth': (3, 8.99), 'alpha': (0, 5), 'lambda': (0, 3),
    'min_child_weight': (5, 50), 'gamma': (0, 10), 'eta': (0, 0.3),
}


def lgb_mse():
    lgb_train, lgb_test = ModelRun(n_fold=10).run_oof(
        lgbwrapper_mae, x_train.values, y_train.values, x_test.values, clf2=lgbwrapper_mse)
    return lgb_train, lgb_test


def lgb_bayes_mae():
    # Bayes MAE
    lgbwrapper_mae.optimize(x_train.values, y_train.values, param_grid=lgb_adj_params, eval_func='mae')


def lgb_logy_bayes_mae():
    global y_train
    logy_train = log_y_value(y_train)
    # Bayes MAE
    lgbwrapper_mae.optimize(x_train.values, logy_train.values, param_grid=lgb_adj_params, eval_func='mae')


def lgb_bayes_mse():
    # Bayes MSE
    lgbwrapper_mse.optimize(x_train.values, y_train.values, param_grid=lgb_adj_params, eval_func='rmse')


def lgb_logy_bayes_mse():
    global y_train
    logy_train = log_y_value(y_train)
    # Bayes MAE
    lgbwrapper_mse.optimize(x_train.values, logy_train.values, param_grid=lgb_adj_params, eval_func='rmse')


def xgb_mse():
    xgb_train, xgb_test = ModelRun(n_fold=10).run_oof(
        xgbwrapper_mae, x_train.values, y_train.values, x_test.values, clf2=xgbwrapper_mse)
    return xgb_train, xgb_test


def xgb_bayes_mae():
    # Bayes mae
    xgbwrapper_mae.optimize(x_train.values, y_train.values, param_grid=xgb_adj_params, eval_func='mae')


def xgb_bayes_mse():
    # Bayes mse
    xgbwrapper_mse.optimize(x_train.values, y_train.values, param_grid=xgb_adj_params, eval_func='rmse')


if __name__ == '__main__':
    train_test = get_feature(train_test)
    x_train, y_train, x_test = train_test[:train_df.shape[0]], y_train, train_test[train_df.shape[0]:]

    # lgb
    lgbwrapper_mae = GbmWrapper(LGBMRegressor, params=lgb_params_mae)
    lgbwrapper_mse = GbmWrapper(LGBMRegressor, params=lgb_params_mse, eval_metric='rsme')

    #####  XGB  ####
    xgbwrapper_mae = XgbWrapper(xgb_params_mae)
    xgbwrapper_mse = XgbWrapper(xgb_params_mse)

    if len(sys.argv) <= 1:
        print('python {} lgb_mse|lgb_bayes_mae|lgb_bayes_mse|xgb_mse|xgb_bayes_mae|xgb_bayes_mse'.format(sys.argv[0]))
    else:
        names = sys.argv[1:]
        for n in names:
            if n in ['lgb_mse', 'xgb_mse']:
                tr, te = globals()[n]()
                df = pd.DataFrame({'train_pred': tr, 'test_pred': te})
                df.to_csv('{}.res.csv'.format(n), index=False)
            else:
                globals()[n]()
