import os.path
import pathlib

import matplotlib.pyplot as plt
import pandas as pd

import definitions


def test_compare_rain_by_pics():
    origin_rain_pics = os.path.join(definitions.ROOT_DIR, 'example/biliu_history_data/origin_biliu_pics')
    filtered_by_time_pics = os.path.join(definitions.ROOT_DIR, 'example/filtered_rain_pics_by_time')
    filtered_by_space_pics = os.path.join(definitions.ROOT_DIR, 'example/filtered_rain_pics_by_space')
    biliu_rain_path = os.path.join(definitions.ROOT_DIR, 'example/biliu_history_data/history_data_splited_hourly')
    for dir_name, sub_dirs, files in os.walk(biliu_rain_path):
        for file in files:
            if '水位' not in file:
                csv_path = os.path.join(biliu_rain_path, file)
                df = pd.read_csv(csv_path, engine='c')
                df.plot(x='systemtime', y='paravalue', xlabel='time', ylabel='rain')
                plt.savefig(os.path.join(origin_rain_pics, file.split('.')[0]+'.png'))
    sl_rain_path = os.path.join(definitions.ROOT_DIR, 'example/rain_datas')
    for dir_name, sub_dirs, files in os.walk(sl_rain_path):
        for file in files:
            csv_path = os.path.join(sl_rain_path, file)
            df = pd.read_csv(csv_path, engine='c')
            df.plot(x='TM', y='DRP', xlabel='time', ylabel='rain')
            plt.savefig(os.path.join(origin_rain_pics, file.split('.')[0]+'.png'))
    total_rain_filtered_by_time_path = os.path.join(definitions.ROOT_DIR, 'example/filtered_data_by_time')
    for dir_name, sub_dirs, files in os.walk(total_rain_filtered_by_time_path):
        for file in files:
            csv_path = os.path.join(total_rain_filtered_by_time_path, file)
            df = pd.read_csv(csv_path, engine='c')
            if ('systemtime' in df.columns) & ('paravalue' in df.columns):
                df.plot(x='systemtime', y='paravalue', xlabel='time', ylabel='rain')
            elif ('TM' in df.columns) & ('DRP' in df.columns):
                df.plot(x='TM', y='DRP', xlabel='time', ylabel='rain')
            plt.savefig(os.path.join(filtered_by_time_pics, file.split('.')[0]+'.png'))
    total_rain_filtered_by_space_path = os.path.join(definitions.ROOT_DIR, 'example/filtered_rain_between_sl_biliu')
    for dir_name, sub_dirs, files in os.walk(total_rain_filtered_by_space_path):
        for file in files:
            csv_path = os.path.join(total_rain_filtered_by_space_path, file)
            df = pd.read_csv(csv_path, engine='c')
            plt.xlabel('time')
            plt.ylabel('rain')
            if ('systemtime' in df.columns) & ('paravalue' in df.columns):
                df.plot(x='systemtime', y='paravalue', xlabel='time', ylabel='rain')
            elif ('TM' in df.columns) & ('DRP' in df.columns):
                df.plot(x='TM', y='DRP', xlabel='time', ylabel='rain')
            plt.savefig(os.path.join(filtered_by_space_pics, file.split('_')[0]+'.png'))


