import pandas as pd
import numpy as np


def from_excel(file_name,):
    data = pd.read_excel(file_name)
    header = data.columns
    header = pd.Series(header[0].split(','))
    header[47] = 't28'
    
    value = data.values
    value_len = len(value[0][0].split(','))
    value_array = np.empty((value.shape[0], value_len), dtype='U40')
    for i, t in enumerate(value):
        for j in range(value_len):
            value_array[i, j] = t[0].split(',')[j]
            
    data = pd.DataFrame(value_array, columns=header).set_index([''])
    data = data.drop('ID', axis=1).astype(np.float)
    
    return data


def make_data(data, target,):
    features = data.drop([target], axis=1)
    target = data[target]
    return features, target


def make_feature(features, target,):
    f = features.T.drop_duplicates().T
    variance = f.var()
    
    for col in f:
        var = variance[col]
        if var < 0.001:
            f = f.drop([col], axis=1)
    
    corr = f.corr(method='spearman')
    depended_val = []
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            if i < j:
                if corr.iloc[i][j] > 0.8:
                    if j not in depended_val:
                        depended_val.append(j)
    final = f.drop(f.columns[depended_val], axis=1)

    return final
