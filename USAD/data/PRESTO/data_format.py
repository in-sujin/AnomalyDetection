import os
import glob
import pickle
import numpy as np
import matplotlib.pylab as plt
import scipy.io
import pandas as pd

def mat_xz_yz(path):
    files = glob.glob(path + "/*.mat")
    files.sort()
    
    xz_mats = []
    yz_mats = []
    
    for file in files:
        if file.find("XZ") != -1:
            mat = scipy.io.loadmat(file)
            xz_mats.append(mat)
        elif file.find("YZ") != -1:
            mat = scipy.io.loadmat(file)
            yz_mats.append(mat)
    
    return xz_mats, yz_mats

def add_data_to_dataframe(p1_xz, p1_yz, p2_xz, p2_yz, df):
    new_data = {
        'p1_xz_ho': p1_xz['yDataCurrentAxes1_tim'].flatten(),
        'p1_xz_ve': p1_xz['yDataCurrentAxes2_tim'].flatten(),
        'p1_yz_ho': p1_yz['yDataCurrentAxes1_tim'].flatten(),
        'p1_yz_ve': p1_yz['yDataCurrentAxes2_tim'].flatten(),
        'p2_xz_ho': p2_xz['yDataCurrentAxes1_tim'].flatten(),
        'p2_xz_ve': p2_xz['yDataCurrentAxes2_tim'].flatten(),
        'p2_yz_ho': p2_yz['yDataCurrentAxes1_tim'].flatten(),
        'p2_yz_ve': p2_yz['yDataCurrentAxes2_tim'].flatten()
    }
    
    new_df = pd.DataFrame(new_data)
    df = pd.concat([df, new_df], ignore_index=True)
    return df

def process_data(point1_path, point2_path):
    point1_xz_mats, point1_yz_mats = mat_xz_yz(point1_path)
    point2_xz_mats, point2_yz_mats = mat_xz_yz(point2_path)
    
    df = pd.DataFrame(columns=['p1_xz_ho', 'p1_xz_ve', 'p1_yz_ho', 'p1_yz_ve', 'p2_xz_ho', 'p2_xz_ve', 'p2_yz_ho', 'p2_yz_ve'])

    for idx in range(len(point1_xz_mats)):
        df = add_data_to_dataframe(point1_xz_mats[idx], point1_yz_mats[idx], point2_xz_mats[idx], point2_yz_mats[idx], df)
    
    return df

def save_data(df, otuput_name, save_file_format):
    output_path = output_name + '.' + save_file_format
    if save_file_format == 'csv':
        df.to_csv(output_path, index=False)
        print("*** csv file save : ", output_path)
        
    elif save_file_format == 'pkl':
        df.to_pickle(output_path)
        print("*** pkl file save : ", output_path)
        
    elif save_file_format == 'mat':
        scipy.io.savemat(output_path, {
            'p1_xz_ho': df['p1_xz_ho'],
            'p1_xz_ve': df['p1_xz_ve'],
            'p1_yz_ho': df['p1_yz_ho'],
            'p1_yz_ve': df['p1_yz_ve'],
            'p2_xz_ho': df['p2_xz_ho'],
            'p2_xz_ve': df['p2_xz_ve'],
            'p2_yz_ho': df['p2_yz_ho'],
            'p2_yz_ve': df['p2_yz_ve']
        })
        print("*** mat file save : ", output_path)
    else:
        print(" save file format error! ")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 5:
        print("Usage: python script.py <path_to_point1_data> <path_to_point2_data> <output_name> <save_file_format>")
        sys.exit(1)
    
    point1_path = sys.argv[1]
    point2_path = sys.argv[2]
    output_name = sys.argv[3]
    file_format = sys.argv[4]

    df = process_data(point1_path, point2_path)
    save_data(df, output_name, file_format)
    
