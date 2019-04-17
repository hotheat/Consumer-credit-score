import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_absolute_error
import time
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import seaborn as sns


class Null_Importance(object):
    def __init__(self, x_train, y_train):
        self.eval_func = mean_absolute_error
        self.x_train = x_train
        self.y_train = y_train
        self.data = pd.concat((self.x_train, self.y_train), axis=1)

    def get_feature_importances(self, shuffle=False, seed=None):
        # Gather real features
        train_features = list(self.x_train.columns)
        # Go over fold and keep track of CV score (train and valid) and feature importances

        # Shuffle target if required
        y = self.y_train.copy()
        if shuffle:
            # Here you could as well use a binomial distribution
            y = self.y_train.copy().sample(frac=1.0)

        # Fit LightGBM in RF mode, yes it's quicker than sklearn RandomForest
        dtrain = lgb.Dataset(self.data[train_features], y,
                             free_raw_data=False, silent=True)
        lgb_params = {
            'objective': 'regression',
            'boosting_type': 'rf',
            'subsample': 0.623,
            'colsample_bytree': 0.7,
            'num_leaves': 127,
            'max_depth': 8,
            'seed': seed,
            'bagging_freq': 1,
            'n_jobs': 4
        }

        # Fit the model
        clf = lgb.train(params=lgb_params, train_set=dtrain,
                        num_boost_round=200)

        # Get feature importances
        imp_df = pd.DataFrame()
        imp_df["feature"] = train_features
        imp_df["importance_gain"] = clf.feature_importance(importance_type='gain')
        imp_df["importance_split"] = clf.feature_importance(
            importance_type='split')
        imp_df['trn_score'] = self.eval_func(y, clf.predict(self.data[train_features]))
        return imp_df

    def build_null_distribution(self):
        null_imp_df = pd.DataFrame()
        nb_runs = 80
        start = time.time()
        dsp = ''
        for i in range(nb_runs):
            # Get current run importances
            imp_df = self.get_feature_importances(shuffle=True)
            imp_df['run'] = i + 1
            # Concat the latest importances with the old ones
            null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)
            # Erase previous message
            for l in range(len(dsp)):
                print('\b', end='', flush=True)
            # Display current run and time used
            spent = (time.time() - start) / 60
            dsp = 'Done with %4d of %4d (Spent %5.1f min)' % (i + 1, nb_runs, spent)
            print(dsp, end='', flush=True)
        return null_imp_df

    def display_distributions(self, actual_imp_df_, null_imp_df_, feature_):
        plt.figure(figsize=(13, 6))
        gs = gridspec.GridSpec(1, 2)
        # Plot Split importances
        ax = plt.subplot(gs[0, 0])
        a = ax.hist(null_imp_df_.loc[null_imp_df_['feature'] == feature_, 'importance_split'].values,
                    label='Null importances')
        ax.vlines(x=actual_imp_df_.loc[actual_imp_df_['feature'] == feature_, 'importance_split'].mean(),
                  ymin=0, ymax=np.max(a[0]), color='r', linewidth=10, label='Real Target')
        ax.legend()
        ax.set_title('Split Importance of %s' % feature_.upper(), fontweight='bold')
        plt.xlabel('Null Importance (split) Distribution for %s ' % feature_.upper())
        # Plot Gain importances
        ax = plt.subplot(gs[0, 1])
        a = ax.hist(null_imp_df_.loc[null_imp_df_['feature'] == feature_, 'importance_gain'].values,
                    label='Null importances')
        ax.vlines(x=actual_imp_df_.loc[actual_imp_df_['feature'] == feature_, 'importance_gain'].mean(),
                  ymin=0, ymax=np.max(a[0]), color='r', linewidth=10, label='Real Target')
        ax.legend()
        ax.set_title('Gain Importance of %s' % feature_.upper(), fontweight='bold')
        plt.xlabel('Null Importance (gain) Distribution for %s ' % feature_.upper())

    def score(self, actual_imp_df, null_imp_df):
        feature_scores = []
        for _f in actual_imp_df['feature'].unique():
            f_null_imps_gain = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
            f_act_imps_gain = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].mean()
            gain_score = np.log(
                1e-10 + f_act_imps_gain / (1 + np.percentile(f_null_imps_gain, 75)))  # Avoid didvide by zero
            f_null_imps_split = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
            f_act_imps_split = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].mean()
            split_score = np.log(
                1e-10 + f_act_imps_split / (1 + np.percentile(f_null_imps_split, 75)))  # Avoid didvide by zero
            feature_scores.append((_f, split_score, gain_score))

        scores_df = pd.DataFrame(feature_scores, columns=['feature', 'split_score', 'gain_score'])

        plt.figure(figsize=(16, 16))
        gs = gridspec.GridSpec(1, 2)
        # Plot Split importances
        ax = plt.subplot(gs[0, 0])
        sns.barplot(x='split_score', y='feature', data=scores_df.sort_values('split_score', ascending=False).iloc[0:70],
                    ax=ax)
        ax.set_title('Feature scores wrt split importances', fontweight='bold', fontsize=14)
        # Plot Gain importances
        ax = plt.subplot(gs[0, 1])
        sns.barplot(x='gain_score', y='feature', data=scores_df.sort_values('gain_score', ascending=False).iloc[0:70],
                    ax=ax)
        ax.set_title('Feature scores wrt gain importances', fontweight='bold', fontsize=14)
        plt.tight_layout()

    def check_importance_unrelated_feature(self, actual_imp_df, null_imp_df):
        correlation_scores = []
        for _f in actual_imp_df['feature'].unique():
            f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
            f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].values
            gain_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
            f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
            f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].values
            split_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
            correlation_scores.append((_f, split_score, gain_score))

        corr_scores_df = pd.DataFrame(correlation_scores, columns=['feature', 'split_score', 'gain_score'])

        fig = plt.figure(figsize=(16, 16))
        gs = gridspec.GridSpec(1, 2)
        # Plot Split importances
        ax = plt.subplot(gs[0, 0])
        sns.barplot(x='split_score', y='feature',
                    data=corr_scores_df.sort_values('split_score', ascending=False).iloc[0:70], ax=ax)
        ax.set_title('Feature scores wrt split importances', fontweight='bold', fontsize=14)
        # Plot Gain importances
        ax = plt.subplot(gs[0, 1])
        sns.barplot(x='gain_score', y='feature',
                    data=corr_scores_df.sort_values('gain_score', ascending=False).iloc[0:70], ax=ax)
        ax.set_title('Feature scores wrt gain importances', fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.suptitle("Features' split and gain scores", fontweight='bold', fontsize=16)
        fig.subplots_adjust(top=0.93)
