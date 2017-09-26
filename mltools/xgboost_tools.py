import os
import tempfile
import shutil
import json
from time import time

import numpy as np

import xgboost as xgb
from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

class XGBoostParamsOptimizer:
    def _get_metrics_results(self, metrics, cv_result):
        maximize_metrics = ('auc', 'map', 'ndcg')
        metrics_results = {}
        for metric in metrics:
            cv_col = cv_result['test-{}-mean'.format(metric)]
            best_min = metric not in maximize_metrics
            metrics_results[metric] = {'best_value' : cv_col.min() if best_min else cv_col.max(),
                                       'last_value' : cv_col.iloc[-1],
                                       'best_iteration' : int(cv_col.idxmin() if best_min else cv_col.idxmax()) + 1,
                                       'last_iteration' : len(cv_col)}
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
                     show_stdv = False)

        metrics_results = self._get_metrics_results(metrics, res)
        self._results.append((params, metrics_results))
        with open(self._filename, 'w') as f:
            json.dump(self._results, f)

        score = metrics_results[score_metric]['best_value']
        end_time = time()
        print("Score {} on {}-th iteration in {}".format(score, metrics_results[score_metric]['best_iteration'],
                                                         end_time - start_time))
        return {'loss' : score, 'status': STATUS_OK}
    
    def _optimize_trees(self, trials, max_evals, max_depth_min, max_depth_max, min_eta, max_eta, eta_step,
                        objective, score_metric, extra_metrics, zero_min_child_weight, zero_gamma,
                        use_hists, grow_policy, max_numround, scale_pos_weight, num_class):
        
        gamma = 0.0 if zero_gamma else hp.qloguniform('gamma', np.log(1e-5), np.log(1e1), 1)
        min_child_weight = 0.0 if zero_min_child_weight else hp.qloguniform('min_child_weight',
                                                                            np.log(1e-5), np.log(1e1), 1)
        space = {
            'eta' : hp.quniform('eta', min_eta, max_eta, eta_step),
            'max_depth' : hp.quniform('max_depth', max_depth_min, max_depth_max, 1),
            'min_child_weight' : min_child_weight,
            'subsample' : hp.quniform('subsample', 0.5, 1, 0.05),
            'colsample_bytree' : hp.quniform('colsample_bytree', 0.5, 1, 0.05),
            'objective': objective,
            'gamma' : gamma,
            'silent' : 1,
            'score_metric' : score_metric,
            'extra_metrics' : extra_metrics,
            'numround' : max_numround,
        }
        if objective.startswith('binary'):
            space['scale_pos_weight'] = scale_pos_weight
        if objective.startswith('multi'):
            space['num_class'] = num_class
        if use_hists:
            space['tree_method'] = 'hist'
            space['grow_policy'] = grow_policy

        best = fmin(self._xgb_binary_score, space, algo=tpe.suggest, trials=trials, max_evals=max_evals)
        print(best)
        
    def get_optimal_xgb_params(self, features, target, folds, binary=True, score_metric=None, extra_metrics=None,
                               max_depth_min=2, max_depth_max=6, min_eta=0.02, max_eta=0.2, eta_step=0.02,
                               max_evals=50, zero_min_child_weight=True, zero_gamma=True, use_hists=False,
                               grow_policy='depthwise', weigh_positives=False, max_numround=1000,
                               task_name='xgb_params', tmp_dir=None, remove_tmp_dir=False):
        if score_metric is not None:
            if binary and score_metric not in ['logloss', 'error', 'rmse', 'auc']:
                raise ValueError('use correct metrics for binary tasks')
            if not binary and score_metric not in ['mlogloss', 'merror']:
                raise ValueError('use correct metrics for multiclass tasks')
        if score_metric is None:
            score_metric = 'logloss' if binary else 'mlogloss'
        objective = 'binary:logistic' if binary else 'multi:softprob'
        num_class = None if binary else len(target.value_counts())
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
        self._folds = folds

        trials = Trials()
        self._optimize_trees(trials, max_evals, max_depth_min, max_depth_max, min_eta, max_eta, eta_step, 
                             objective, score_metric, extra_metrics, zero_min_child_weight, zero_gamma,
                             use_hists, grow_policy, max_numround, scale_pos_weight, num_class)
        
        if remove_tmp_dir:
            shutil.rmtree(self._tmp_dir)
        
        return self._results