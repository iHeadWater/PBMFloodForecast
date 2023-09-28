import os

import pandas as pd

import definitions
from hydromodel.division.dmca_esr import STEP1_STEP2_Tr_and_fluctuations_timeseries, STEP3_core_identification, STEP4_end_rain_events, \
    STEP5_beginning_rain_events


def test_session_division_new():
    rain = pd.read_csv(os.path.join(definitions.ROOT_DIR, 'hydromodel/example/division_rain.csv'))['rain']
    flow = pd.read_csv(os.path.join(definitions.ROOT_DIR, 'hydromodel/example/division_flow.csv'))['flow']
    time = rain.index.to_numpy()
    rain = rain.to_numpy()/24
    flow = flow.to_numpy()/24
    rain_min = 0.02
    max_window = 100
    Tr, fluct_rain_Tr, fluct_flow_Tr, fluct_bivariate_Tr = STEP1_STEP2_Tr_and_fluctuations_timeseries(rain, flow, rain_min, max_window)
    beginning_core, end_core = STEP3_core_identification(fluct_bivariate_Tr)
    end_rain = STEP4_end_rain_events(beginning_core, end_core, rain, fluct_rain_Tr, rain_min)
    beginning_rain = STEP5_beginning_rain_events(beginning_core, end_rain, rain, fluct_rain_Tr, rain_min)
    '''
    print('________________________________')
    print('beginning_core = '+str(beginning_core.T))
    print('________________________________')
    print('end_core = '+str(end_core.T))
    print(len(beginning_core), len(end_core))
    '''
    print('________________________________')
    print('beginning_rain = '+str(beginning_rain.T))
    print('________________________________')
    print('end_rain = '+str(end_rain.T))
    print(len(beginning_rain), len(end_rain))