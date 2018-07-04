import pandas as pd
import os


def args_to_csv(dst_path, config):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    dict_ = vars(config)
    s = pd.Series(data=dict_)
    print('====== Parameters ======')
    print(s)
    print('========================')
    s.to_csv(dst_path)


def csv_to_dict(csv_path):
    s = pd.Series.from_csv(csv_path)
    for k in s.keys():
        # str -> bool
        if s[k] == 'True':
            s[k] = True
            continue
        elif s[k] == 'False':
            s[k] = False
            continue
        # str -> list
        if s[k][0] == '[' and s[k][-1] == ']':
            s[k] = s[k][1:-1].replace("'", '').split(', ')
    return s.to_dict()
