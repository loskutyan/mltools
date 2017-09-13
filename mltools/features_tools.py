import re
import logging
import numpy as np
import pandas as pd
from collections import Counter
from functools import reduce
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

from pymystem3 import Mystem

class DataFrameScaler:
    def __init__(self, scale_ints=True, smart_log=True, log_scale_threshold=1e+3):
        self._scalers = None
        self._scale_ints = scale_ints
        self._smart_log = smart_log
        self._log_cols = set()
        self._biased_cols = {}
        self._log_scale_threshold = log_scale_threshold
            
        self._cols_idx_to_scale = None
        self._cols_to_scale = None
        
    def fit(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError('argument must be pandas DataFrame')
        if self._scale_ints:
            self._cols_idx_to_scale = np.where(np.logical_or(X.dtypes == float, X.dtypes == int))[0]
        else:
            self._cols_idx_to_scale = np.where(X.dtypes == float)[0]
        self._cols_to_scale = X.columns[self._cols_idx_to_scale]
        self._scalers = {col : StandardScaler() for col in self._cols_to_scale}
        for col, scaler in self._scalers.items():
            data_to_scale = X[col].dropna()
            scaler.fit(data_to_scale.values.reshape(-1, 1))
            if self._smart_log and scaler.scale_ > self._log_scale_threshold and not min(data_to_scale) < 0.0:
                if min(data_to_scale) == 0.0:
                    scaler.fit(np.log((data_to_scale + 1.).values.reshape(-1, 1)))
                    self._biased_cols[col] = 1.
                else:
                    scaler.fit(np.log(data_to_scale.values.reshape(-1, 1)))
                self._log_cols.add(col)
                
        
    def transform(self, X):
        result = X.copy()
        for col, scaler in self._scalers.items():
            if col not in X.columns:
                continue
            nan_indexes = set(np.where(X[col].isnull())[0])
            valid_indexes = np.where(~X[col].isnull())[0]
            if col in self._log_cols:
                if col in self._biased_cols:
                    data_to_scale = np.log(X[col] + self._biased_cols[col])
                else:
                    data_to_scale = np.log(X[col])
            else:
                data_to_scale = X[col]
            scaled = pd.Series(scaler.transform(data_to_scale.iloc[valid_indexes].values.reshape(-1, 1)).reshape(1, -1)[0],
                               index=valid_indexes)
            result[col] = np.array([np.nan if i in nan_indexes else scaled.loc[i] for i in range(len(X))])
        result.columns = ['log_' + x if x in self._log_cols else x for x in X.columns]
        return result
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        cols_to_inverse_scale = [x for x in X.columns if x in self._cols_to_scale]
        cols_to_inverse_scale_idx = [np.where(np.array(self._cols_to_scale) == x)[0][0] for x in cols_to_inverse_scale]
        result = X.copy()
        for i, col in enumerate(cols_to_inverse_scale):
            result[col] = X[col] * self._scaler.scale_[cols_to_inverse_scale_idx[i]] + self._scaler.mean_[cols_to_inverse_scale_idx[i]]
        return result
        
class MyOneHotEncoder:
    def __init__(self, min_freq=0, drop_classes=[], na_allowed=True, extra_classes_allowed=True, logger_name=None):
        self._na_allowed = na_allowed
        self._extra_classes_allowed = extra_classes_allowed
        self._lenc = LabelEncoder()
        self._oenc = OneHotEncoder(sparse=False)
        self._min_freq = min_freq
        self._drop_classes = list(drop_classes)
        self._logger_name = logger_name
    
    def fit(self, series):
        if not self._na_allowed and np.sum(series.isnull()) > 0:
            raise ValueError('NA values not allowed')
        if self._na_allowed or self._extra_classes_allowed:
            series_to_fit = series.append(pd.Series([np.nan], index=['---']))
            labeled = self._lenc.fit_transform(series_to_fit.fillna('---').astype('str'))
        else:
            labeled = self._lenc.fit_transform(series.astype('str'))
        
        self._oenc.fit(labeled.reshape(-1, 1))
        if self._min_freq > 0:
            drop_classes = [k for k, v in Counter(labeled).items() if v < self._min_freq]
            self._drop_classes.extend([self._lenc.classes_[i] for i in drop_classes])
        if len(self._drop_classes) > 0:
            lenc_classes = np.array(self._lenc.classes_)
            self._drop_classes = [y[0] for y in sorted([(k, np.where(lenc_classes == k)[0][0])
                                                        for k in list(set(self._drop_classes))],
                                                       key=lambda x: x[1])]
        if '---' in self._drop_classes:
            self._drop_classes.remove('---')
        
    def transform(self, series, name_prefix=True):
        try:
            if not self._na_allowed and np.sum(series.isnull()) > 0:
                raise ValueError('NA values not allowed')
            series_to_transform = series.fillna('---').astype('str')
            extra_classes = set(series_to_transform) - set(self._lenc.classes_) - set(['---'])
            if not self._extra_classes_allowed and len(extra_classes) > 0:
                raise ValueError('extra values in series to transform not allowed')
            series_to_transform[series_to_transform.isin(extra_classes)] = '---'
            if self._logger_name is not None:
                logging.getLogger(self._logger_name).warning('transforming {}: dropped {}'
                                                             .format(series.name,str(sorted(extra_classes))))
            transformed = pd.DataFrame(self._oenc.transform(self._lenc.transform(series_to_transform).reshape(-1, 1)),
                                columns=self._lenc.classes_, index=series.index, dtype=bool)
            if self._na_allowed or self._extra_classes_allowed:
                transformed = transformed.drop('---', 1)
            if len(self._drop_classes) > 0:
                transformed = transformed.drop(self._drop_classes, 1)
            if name_prefix:
                transformed.columns = [series.name + '_' + x for x in transformed.columns]
            return transformed
        except:
            raise ValueError('{}, {}, {}'.format(series.name, self._lenc.classes_, len(self._oenc.active_features_)))
    
    def fit_transform(self, series, name_prefix=True):
        self.fit(series)
        return self.transform(series, name_prefix)

class TextVectorizer:
    def __init__(self, min_ngram, max_ngram, vocabulary=None, count=False, lowercase=False):
        self.min_ngram = min_ngram
        self.max_ngram = max_ngram
        self.vocabulary = vocabulary
        self.count = count
        self.vectorizer = CountVectorizer(ngram_range=(min_ngram, max_ngram), lowercase=lowercase,
                                          vocabulary=vocabulary, token_pattern = r'(?u)\b\w+\b')

    def fit(self, text_col):
        self.vectorizer.fit(text_col.fillna(''))

    def transform(self, text_col):
        transformed = self.vectorizer.transform(text_col.fillna('')).toarray()
        prefix = ('count_' if self.count else '') + text_col.name + ':_'
        features = pd.DataFrame(transformed.astype('bool') if not self.count else transformed,
                                columns=[prefix + x for x in self.vectorizer.get_feature_names()],
                                index=text_col.index)
        return features

    def fit_transform(self, text_col):
        self.fit(text_col)
        return self.transform(text_col)

class Tokenizer:
    def __init__(self):
        self.sentence_divider = re.compile("^[\.+|\!+|\?+|\)+|\(+|\,+|\:+\;+]\s*")
        self.eng_pattern = re.compile('^[a-zA-Z]+$')
        self.ILLEGAL_CHARACTERS_RE = re.compile(r'[\000-\010]|[\013-\014]|[\016-\037]')
        self.sentence_separators = '.!?'
        self.model_file_name = '2015.model'
        self.intab = "012345678"
        self.outtab = "999999999"
        self.digit_trans = str.maketrans(self.intab, self.outtab)
        self.mystem = Mystem()
        self.none_accum = Counter()
        self.bastard_accum = Counter()
        self.whitelist = {}
        self.stopwords = []
        self.token_filter = lambda x: x[3] is not None
        self.concat_no_s = True

    def get_token(self, item):
        if 'analysis' in item.keys() and len(item['analysis']) > 0:
            return (item['analysis'][0]['lex'],
                    item['text'],
                    (lambda x: 'qual' in x.keys() and x['qual'] or None)(item['analysis'][0]),
                    item['analysis'][0]['gr']
                    )
        else:
            return (item['text'].lower().translate(self.digit_trans),
                    item['text'],
                    None,
                    None
                    )

    def tokenize(self, text):
        analyze = self.mystem.analyze(text)
        stack = []
        # Убираем последний item, потому, что mystem вставляет '/n' после всего текста
        for item in analyze[:-1]:
            token = self.get_token(item)
            if token[3] and "сокр" in token[3]:
                stack.append(token)
            else:
                if stack:
                    if token[0].strip() in self.sentence_separators:
                        new_token = stack[0]
                        stack = []
                        yield new_token
                    else:
                        new_token = stack[0]
                        stack = []
                        yield from [new_token, token]
                else:
                    yield token
    
    def fit(self, text):
        if text is None or str(text) == 'nan' or not isinstance(text, str):
            return []
        sentences = []
        sentence = []
        english_collector = []
        for token in self.tokenize(text):
            if self.sentence_divider.findall(token[0]):
                if english_collector:
                    word = ''.join(english_collector).replace('-', '')
                    english_collector = []
                    if len(word) > 1:
                        sentence += [(word, word, False, 'ENG')]
                if len(sentence) > 0:
                    sentences.append(sentence)
                sentence = []
            if token[3] is None:
                self.none_accum[token[0]] += 1
            if token[2] == 'bastard':
                self.bastard_accum[token[0]] += 1
                
            if english_collector:
                if token[0] == '-':
                    english_collector.append(token[0])
                    continue
                elif english_collector[-1] == '-' and self.eng_pattern.findall(token[0]):
                    english_collector.append(token[0])
                    word = ''.join(english_collector)
                    english_collector = []
                    if len(word) > 1:
                        sentence += [(word, word, False, 'ENG')]
                    continue
                else:
                    word = ''.join(english_collector).replace('-', '')
                    english_collector = []
                    if len(word) > 1:
                        sentence += [(word, word, False, 'ENG')]
            else:
                if self.eng_pattern.findall(token[0]):
                    english_collector.append(token[0])
                    continue

            if (self.token_filter(token) or token[0] in self.whitelist) and token[0] not in self.stopwords:
                sentence += [(token[0], self.ILLEGAL_CHARACTERS_RE.sub("", token[1]).strip().lower(),
                              False, token[3].split('=')[0].split(',')[0] if token[3] is not None else None)]
                if self.concat_no_s and len(sentence[-2:-1]) > 0 and "не" in sentence[-2:-1][0]:
                    sentence[-2:] = [reduce(lambda x, y: (x[0] + y[0], x[1] + y[1], True, y[3]), sentence[-2:])]

        if len(sentence) > 0:
            sentences.append(sentence)

        return sentences


'''
    "в", "с", "и", "к", "а", "о", "я", "у",
    "по", "на", "бы", "но", "ни", "за", "ты", "он", "ко", "со", "до", "то", "от", "из", "же",
    "для", "при", "для", "что", "под", "это", "как", "или", "где", "тот", "так", "кто", "чем", "еще",
    "этот", "туда",
    "нужно", "чтобы",
    "который",

    "т", "х", "д", "р", "м",
    "когда", "либо", "из-за", "банк", "хотя", # "всегда",
    "просто",
]
'''
