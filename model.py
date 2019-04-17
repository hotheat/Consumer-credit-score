import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
import lightgbm as lgb


def log_y_value(y):
    return np.log1p(1 / y * 1e6) * 100


def reverse_logy(logy):
    return 1e6 / np.expm1(logy / 100)


class ModelRun(object):
    def __init__(self, n_fold=5, eval_fun=mean_absolute_error):
        self.n_fold = n_fold
        self.kf = StratifiedKFold(n_splits=self.n_fold, shuffle=False, random_state=2019)
        self.eval_fun = eval_fun

    @staticmethod
    def merge_predictions(pred1, pred2):
        pred_df = pd.DataFrame({'predmae': pred1, 'predmse': pred2})
        pred_df['rank'] = np.argsort(pred_df['predmae'])
        pred_df['pred'] = pred_df['predmae']
        fisrt_n = int(0.3 * len(pred_df))
        last_n = int(0.7 * len(pred_df))

        pred_df['pred'] = np.where((pred_df['rank'] < fisrt_n) | (pred_df['rank'] > last_n),
                                   pred_df['predmae'] * 0.6 + pred_df['predmse'] * 0.4,
                                   pred_df['predmae'])
        return pred_df['pred'].values

    def run_oof(self, clf, X_train, y_train, X_test, clf2=None, logy=False):
        print(clf.get_name())
        preds_train = np.zeros((len(X_train)), dtype=np.float)
        preds_test = np.zeros((len(X_test)), dtype=np.float)
        train_loss = []
        test_loss = []
        i = 1
        for train_index, test_index in self.kf.split(X_train, y_train):
            x_tr = X_train[train_index]
            x_te = X_train[test_index]
            y_tr = y_train[train_index]
            y_te = y_train[test_index]

            if clf.get_name() in ['lightGBM', 'XGBoost']:
                if logy:
                    y_tr, y_te = log_y_value(y_tr), log_y_value(y_te)
                clf.fit(x_tr, y_tr, x_te, y_te)
                x_tr_pred, x_te_pred = clf.predict(x_tr), clf.predict(x_te)
                if logy:
                    y_tr, y_te, x_tr_pred, x_te_pred = reverse_logy(y_tr), reverse_logy(y_te), reverse_logy(
                        x_tr_pred), reverse_logy(x_te_pred)
                loss_tr, loss_te = self.eval_fun(y_tr, x_tr_pred), \
                                   self.eval_fun(y_te, x_te_pred)

                X_test_pred = clf.predict(X_test)
                if logy:
                    X_test_pred = reverse_logy(X_test_pred)

                if clf2 is not None:
                    if logy:
                        y_tr, y_te = log_y_value(y_tr), log_y_value(y_te)
                    clf2.fit(x_tr, y_tr, x_te, y_te)
                    x_tr_pred2, x_te_pred2 = clf2.predict(x_tr), clf2.predict(x_te)

                    if logy:
                        y_tr, y_te, x_tr_pred2, x_te_pred2 = reverse_logy(y_tr), reverse_logy(y_te), reverse_logy(
                            x_tr_pred2), reverse_logy(x_te_pred2)

                    x_tr_pred_merge = ModelRun.merge_predictions(x_tr_pred, x_tr_pred2)
                    x_te_pred_merge = ModelRun.merge_predictions(x_te_pred, x_te_pred2)

                    print('mae 预测', x_te_pred[x_te_pred != x_te_pred_merge][:5])
                    print('mse 预测', x_te_pred2[x_te_pred != x_te_pred_merge][:5])
                    print('合并后的预测', x_te_pred_merge[x_te_pred != x_te_pred_merge][:5])
                    print('y 实际值', y_te[x_te_pred != x_te_pred_merge][:5])

                    loss_tr, loss_te = self.eval_fun(y_tr, x_tr_pred_merge), \
                                       self.eval_fun(y_te, x_te_pred_merge)

                    X_test_pred2 = clf2.predict(X_test)

                    if logy:
                        X_test_pred2 = reverse_logy(X_test_pred2)

                    X_test_pred = ModelRun.merge_predictions(X_test_pred, X_test_pred2)

                train_loss.append(1 / (1 + loss_tr))
                test_loss.append(1 / (1 + loss_te))
                preds_train[test_index] = x_te_pred
                preds_test += X_test_pred
            else:
                clf.fit(x_tr, y_tr)
                loss_tr = self.eval_fun(y_tr, clf.predict(x_tr))
                loss_te = self.eval_fun(y_te, clf.predict(x_te))
                train_loss.append(1 / (1 + loss_tr))
                test_loss.append(1 / (1 + loss_te))

                preds_train[test_index] = clf.predict(x_te)
                preds_test += clf.predict(X_test)

            print('{0}: Train {1:0.7f} Val {2:0.7f}/{3:0.7f}'.format(i, train_loss[-1], test_loss[-1],
                                                                     np.mean(test_loss)))
            print('-' * 50)
            i += 1
        print('Train: ', train_loss)
        print('Val: ', test_loss)
        print('-' * 50)
        print('{0} Train{1:0.5f}_Test{2:0.5f}\n\n'.format(clf.get_name(), np.mean(train_loss), np.mean(test_loss)))
        preds_test /= self.n_fold
        return preds_train, preds_test


class SklearnWrapper(object):
    def __init__(self, clf, params=None):
        if params:
            self.clf = clf(**params)
        else:
            self.clf = clf

    def fit(self, X_train, y_train, X_val=None, y_val=None, eval_func=None):
        self.clf.fit(X_train, y_train)

    def predict_proba(self, x):
        proba = self.clf.predict_proba(x)
        return proba

    def predict(self, x):
        return self.clf.predict(x)

    def optimize(self, X_train, y_train, X_eval, y_eval, param_grid, eval_func, is_proba):
        def fun(**params):
            for k in params:
                if type(param_grid[k][0]) is int:
                    params[k] = int(params[k])

            self.clf.set_params(**params)
            self.fit(X_train, y_train)

            if is_proba:
                p_eval = self.predict_proba(X_eval)
            else:
                p_eval = self.predict(X_eval)

            best_score = eval_func(y_eval, p_eval)

            return -best_score

        opt = BayesianOptimization(fun, param_grid)
        opt.maximize(n_iter=100)

        print("Best mae: %.5f, params: %s" % (opt.res['max']['max_val'], opt.res['max']['max_params']))

    def get_name(self):
        return self.clf.__class__.__name__

    # def report(self, y, y_pred, y_pred_proba):
    #     print(self.get_name() + ' report：\n', classification_report(y, y_pred))
    #     print(self.get_name() + ' AUC：\n', roc_auc_score(y, y_pred_proba))


class XgbWrapper(object):
    def __init__(self, params):
        self.param = params

    def fit(self, X_train, y_train, X_val=None, y_val=None, num_round=100000, feval=None):
        dtrain = xgb.DMatrix(X_train, label=y_train)

        if X_val is None:
            watchlist = [(dtrain, 'train')]
        else:
            deval = xgb.DMatrix(X_val, label=y_val)
            watchlist = [(dtrain, 'train'), (deval, 'val')]

        if feval is None:
            self.clf = xgb.train(self.param, dtrain, num_round, evals=watchlist, verbose_eval=200,
                                 early_stopping_rounds=100, )
        else:
            self.clf = xgb.train(self.param, dtrain, num_round, evals=watchlist, feval=feval, verbose_eval=200,
                                 early_stopping_rounds=100)

    def predict_proba(self, x):
        return self.clf.predict(xgb.DMatrix(x))

    def predict(self, x):
        return self.clf.predict(xgb.DMatrix(x))

    def optimize(self, X_train, y_train, param_grid, eval_func=None, is_proba=False, seed=42):
        bayes_eval = {'mae': 'test-mae-mean', 'rmse': 'test-rmse-mean'}
        eval_f = {'mae': mean_absolute_error, 'rmse': mean_squared_error}
        feval = lambda y_pred, y_true: ('mae', eval_f[eval_func](y_true.get_label(), y_pred))
        seed = 2019
        # prepare data
        dtrain = xgb.DMatrix(X_train, label=y_train)

        def fun(**kw):
            params = self.param.copy()
            params['seed'] = seed

            for k in kw:
                if type(param_grid[k][0]) is int:
                    params[k] = int(kw[k])
                else:
                    params[k] = kw[k]
            model = xgb.cv(params, dtrain, 100000, 5, feval=feval, verbose_eval=None, early_stopping_rounds=100)
            return (-1.0 * np.array(model[bayes_eval[eval_func]])).max()

        opt = BayesianOptimization(fun, param_grid)
        opt.maximize(n_iter=100)
        max_df = pd.DataFrame(opt.max)
        print(max_df)
        max_df.to_csv('xgb_bayes_{}.csv'.format(eval_func))

    def get_name(self):
        return 'XGBoost'


class LRWrapper(object):
    def __init__(self, params):
        self.param = params

    def fit(self, X_train, y_train, X_val=None, y_val=None, num_round=100000, feval=None):
        #### 对数据往往先做一步均一化, z-score
        self.clf = Pipeline([('sc', StandardScaler()),
                             ('clf', LogisticRegression())])
        self.clf.fit(X_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def get_name(self):
        return 'Logistic Regression'


class RgfWrapper(object):
    def __init__(self, clf, params=None):
        if params:
            self.param = params
            self.clf = clf(**params)
        else:
            self.clf = clf

    def fit(self, X_train, y_train, X_val=None, y_val=None, eval_func=None):
        self.clf.fit(X_train, y_train)

    def predict_proba(self, x):
        return self.clf.predict_proba(x)

    def predict(self, x):
        return self.clf.predict(x)

    def get_name(self):
        return 'RegularizedGreedyForest'


class GbmWrapper(object):
    def __init__(self, clf, params=None, eval_metric='mae'):
        if params:
            self.param = params
            self.clf = clf(**params)
        else:
            self.clf = clf
        GbmWrapper.eval_metric = eval_metric

    def fit(self, X_train, y_train, X_val, y_val):
        self.clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric=GbmWrapper.eval_metric,
                     early_stopping_rounds=200,
                     verbose=False)

    def predict_proba(self, x):
        return self.clf.predict_proba(x)

    def predict(self, x):
        return self.clf.predict(x)

    def get_name(self):
        return 'lightGBM'

    # TODO
    def optimize(self, X_train, y_train, param_grid, eval_func, is_proba=False):
        seed = 2019
        bayes_eval = {'mae': 'l1-mean', 'rmse': 'rmse-mean'}
        # prepare data
        train_data = lgb.Dataset(data=X_train, label=y_train, free_raw_data=False)

        def fun(**kw):
            params = self.param.copy()
            params['seed'] = seed

            for k in kw:
                if type(param_grid[k][0]) is int:
                    params[k] = int(kw[k])
                else:
                    params[k] = kw[k]

            model = lgb.cv(params, train_data, nfold=5, seed=2019, stratified=False,
                           verbose_eval=200, metrics=[eval_func])

            return (-1.0 * np.array(model[bayes_eval[eval_func]])).max()

        opt = BayesianOptimization(fun, param_grid)
        opt.maximize(n_iter=100)
        max_df = pd.DataFrame(opt.max)
        print(max_df)
        max_df.to_csv('lgb_bayes_{}.csv'.format(eval_func))
