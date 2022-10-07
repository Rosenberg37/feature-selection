# -*- coding: UTF-8 -*-
"""
@Project:   feature-selection 
@File:      read.py
@Author:    Rosenberg
@Date:      2022/10/7 18:47 
@Documentation: 
    ...
"""
import os


def get_csv_paths(data_path: str):
    csv_paths = list()
    for path in os.listdir(data_path):
        cur_path = os.path.join(data_path, path)
        if os.path.isdir(cur_path):
            csv_paths += get_csv_paths(cur_path)
        elif cur_path.endswith('.csv'):
            csv_paths.append(cur_path)
    return csv_paths
