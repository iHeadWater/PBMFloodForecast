import os
from datetime import datetime

import cdsapi
import calendar

import numpy as np
import pandas as pd
import xarray as xr

import definitions

c = cdsapi.Client()  # 创建用户

# 数据信息字典
dic = {
    'product_type': 'reanalysis-era5-land',  # 产品类型
    'format': 'netcdf',  # 数据格式
    'variable': 'potential_evaporation',  # 变量名称
    'year': '',  # 年，设为空
    'month': '',  # 月，设为空
    'day': [],  # 日，设为空
    'time': [  # 小时
        '00:00', '01:00', '02:00', '03:00', '04:00', '05:00',
        '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
        '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
        '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'
    ],
    'area': [41, 122, 39, 123]
}


def test_download_era5():
    # 通过循环批量下载1979年到2020年所有月份数据
    for y in range(2013, 2024):  # 遍历年
        for m in range(1, 13):  # 遍历月
            day_num = calendar.monthrange(y, m)[1]  # 根据年月，获取当月日数
            # 将年、月、日更新至字典中
            dic['year'] = str(y)
            dic['month'] = str(m).zfill(2)
            dic['day'] = [str(d).zfill(2) for d in range(1, day_num + 1)]
            filename = os.path.join(definitions.ROOT_DIR, 'example/era5_data/', 'era5_datas_' + str(y) + str(m).zfill(2) + '.nc')  # 文件存储路径
            c.retrieve('reanalysis-era5-land', dic, filename)


def test_average_pet():
    path = os.path.join(definitions.ROOT_DIR, 'example/era5_data/')
    ds = xr.open_dataset(path + 'era5/201201.nc')
    lats = ds['latitude'].values
    lons = ds['longitude'].values
    grid = pd.read_excel(path + 'grid.xlsx')
    test = pd.DataFrame({'num': range(1, 34), 'latitude': np.nan, 'longitude': np.nan})
    test = test.astype(float)
    for i in range(len(grid['lat'])):
        test['latitude'][i] = np.where(lats == grid['lat'][i])[0][0]
        test['longitude'][i] = np.where(lons == grid['lon'][i])[0][0]
    filenames = ['era5/' + f for f in os.listdir(path + 'era5/')]
    pev_ = pd.DataFrame({'num': range(1, 96433), 'pev': np.nan})
    pev_ = pev_.astype(float)
    for i in range(len(test['num'])):
        sum_time = 0
        for f in filenames:
            ds = xr.open_dataset(path + f)
            pev = ds['pev'].values
            times = pd.to_datetime(ds['time'].values * 3600, origin='1900-01-01')
            for t in range(len(times)):
                pev_[t + sum_time]['pev'] = pev[test['longitude'][i], test['latitude'][i], t] * -100
            pev_[pev_ < 0] = 0
            sum_time += len(times)
        pev_.to_excel(path + 'pet_calc/pev_' +
                      str(grid['lat'][i]) + '_' + str(grid['lon'][i]) + '.xlsx')
    start = datetime(2012, 1, 1, 8)
    end = datetime(2023, 1, 1, 7)
    timesteps = pd.date_range(start, end, freq='H')
    pre = pd.DataFrame(index=timesteps, columns=range(1, 34))
    for i in range(len(grid['FID'])):
        data = pd.read_excel(path + 'pet_calc/pev_' +
                             str(grid['lat'][i]) + '_' + str(grid['lon'][i]) + '.xlsx')
        pre.iloc[:, i] = data['pev']
    pev_jieguo = pd.DataFrame({'time': timesteps})
    pev_jieguo['pre'] = pre.mean(axis=1)
    pev_jieguo.to_excel(path + 'pet_calc/PET_result.xlsx')
