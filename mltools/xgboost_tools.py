import json
import numbers
import os
import shutil
import tempfile
from datetime import datetime
from time import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import ShuffleSplit
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


class XGBoostParamsOptimizer:
    @staticmethod
    def _get_metrics_results(metrics, cv_result):
        maximize_metrics = ('auc', 'map', 'ndcg')
        metrics_results = {}
        for metric in metrics:
            cv_col = cv_result['test-{}-mean'.format(metric)]
            best_min = metric not in maximize_metrics
            metrics_results[metric] = {'best_value': cv_col.min() if best_min else cv_col.max(),
                                       'last_value': cv_col.iloc[-1],
                                       'best_iteration': int(cv_col.idxmin() if best_min else cv_col.idxmax()) + 1,
                                       'last_iteration': len(cv_col)}
        return metrics_results

    def _xgb_binary_score(self, params):
        params['max_depth'] = int(params['max_depth'])

        score_metric = params['score_metric']
        extra_metrics = params['extra_metrics']
        numround = int(params['numround'])
        del params['score_metric']
        del params['extra_metrics']
        del params['numround']

        metrics = sorted({score_metric}.union(set(extra_metrics)), reverse=True)

        start_time = time()
        res = xgb.cv(params, self._dm, numround, folds=self._folds, metrics=metrics, early_stopping_rounds=20,
                     show_stdv=False)

        metrics_results = self._get_metrics_results(metrics, res)
        self._results.append((params, metrics_results))
        with open(self._filename, 'w') as f:
            json.dump(self._results, f)

        score = metrics_results[score_metric]['best_value']
        end_time = time()
        print("Score {} on {}-th iteration in {}".format(score, metrics_results[score_metric]['best_iteration'],
                                                         end_time - start_time))
        return {'loss': score, 'status': STATUS_OK}

    def _optimize_trees(self, trials, max_evals, max_depth_min, max_depth_max, min_eta, max_eta, eta_step,
                        min_lambda, max_lambda, min_alpha, max_alpha, zero_alpha, default_lambda,
                        objectives, score_metric, extra_metrics, zero_min_child_weight, zero_gamma,
                        max_numround, scale_pos_weight, num_class):

        gamma = 0.0 if zero_gamma else hp.loguniform('gamma', np.log(1e-6), np.log(1e3))
        min_child_weight = 0.0 if zero_min_child_weight else hp.loguniform('min_child_weight', np.log(1e-6),
                                                                           np.log(1e3))
        lamb = 1.0 if default_lambda else hp.loguniform('lambda', np.log(min_lambda), np.log(max_lambda))
        alpha = 0.0 if zero_alpha else hp.loguniform('alpha', np.log(min_alpha), np.log(max_alpha))
        space = {
            'eta': hp.quniform('eta', min_eta, max_eta, eta_step),
            'max_depth': hp.quniform('max_depth', max_depth_min, max_depth_max, 1),
            'min_child_weight': min_child_weight,
            'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
            'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
            'objective': hp.choice('objective', objectives),
            'gamma': gamma,
            'alpha': alpha,
            'lambda': lamb,
            'silent': 1,
            'score_metric': score_metric,
            'extra_metrics': extra_metrics,
            'numround': max_numround,
        }
        if objectives[0].startswith('binary'):
            space['scale_pos_weight'] = scale_pos_weight
        if objectives[0].startswith('multi'):
            space['num_class'] = num_class

        best = fmin(self._xgb_binary_score, space, algo=tpe.suggest, trials=trials, max_evals=max_evals)
        print(best)

    def get_optimal_exact_xgb_params(self, features, target, objectives, score_metric, split=0.3, folds=None,
                                     extra_metrics=None, max_depth_min=2, max_depth_max=6, min_eta=0.02, max_eta=0.2,
                                     eta_step=0.02, min_lambda=1.0, max_lambda=1.0, min_alpha=1.0, max_alpha=1.0,
                                     max_evals=50, zero_min_child_weight=True, zero_gamma=True, zero_alpha=True,
                                     default_lambda=True, weigh_positives=False, max_numround=1000,
                                     task_name='xgb_params', tmp_dir=None, remove_tmp_dir=True):
        if extra_metrics is None:
            extra_metrics = []
        if isinstance(objectives, str):
            objectives = [objectives]

        num_class = len(target.value_counts()) if objectives[0].startswith('multi') else None
        positive_number = np.sum(target)
        scale_pos_weight = (len(target) - positive_number) / positive_number if weigh_positives else 1.0

        if tmp_dir is None:
            self._tmp_dir = tempfile.mkdtemp()
        else:
            if not os.path.isdir(tmp_dir):
                os.mkdir(tmp_dir)
            self._tmp_dir = tmp_dir
        self._filename = '{}/{}.json'.format(self._tmp_dir, task_name)
        try:
            with open(self._filename, 'r') as f:
                self._results = json.load(f)
                print('{}: {} previous results read'.format(task_name, len(self._results)))
        except:
            self._results = []
            print('{}: no previous results or error'.format(task_name))

        self._dm = xgb.DMatrix(features, target)

        if folds is None:
            if isinstance(split, datetime) or isinstance(split, pd.Timestamp):
                mask = features.index < split
                folds = [(np.where(mask)[0], np.where(~mask)[0])]
            elif isinstance(split, numbers.Number):
                folds = list(ShuffleSplit(n_splits=1, test_size=split, random_state=0).split(features))
            else:
                raise ValueError('unsupported split type')
        self._folds = folds

        trials = Trials()
        self._optimize_trees(trials, max_evals, max_depth_min, max_depth_max, min_eta, max_eta, eta_step,
                             min_lambda, max_lambda, min_alpha, max_alpha, zero_alpha, default_lambda,
                             objectives, score_metric, extra_metrics, zero_min_child_weight, zero_gamma,
                             max_numround, scale_pos_weight, num_class)

        if remove_tmp_dir:
            shutil.rmtree(self._tmp_dir)

        return self._results
