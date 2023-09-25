import os

import pandas as pd

import definitions
from hydromodel.division.session_division import step1_2_tr_and_fluctuations_timeseries, step3_core_identification, \
    step4_end_rain_events, step5_beginning_rain_events, step6_check_rain_events, step7_end_flow_events, \
    step8_beginning_flow_events, step9_checks_on_flow_events, step10_check_overlapping_events


def test_session_division():
    rain = pd.read_csv(os.path.join(definitions.ROOT_DIR, 'hydromodel/example/division_rain.csv'))['rain']
    flow = pd.read_csv(os.path.join(definitions.ROOT_DIR, 'hydromodel/example/division_flow.csv'))['flow']
    time = rain.index.to_numpy()
    rain = rain.to_numpy()
    flow = flow.to_numpy()
    rain_min = 0.01
    max_window = 100
    Tr, fluct_rain_Tr, fluct_flow_Tr, fluct_bivariate_Tr = step1_2_tr_and_fluctuations_timeseries(rain, flow, rain_min, max_window)
    beginning_core, end_core = step3_core_identification(fluct_bivariate_Tr)
    end_rain = step4_end_rain_events(beginning_core, end_core, rain, fluct_rain_Tr, rain_min)
    beginning_rain = step5_beginning_rain_events(beginning_core, end_rain, rain, fluct_rain_Tr, rain_min)
    beginning_rain_checked, end_rain_checked = step6_check_rain_events(beginning_rain, end_rain, rain, rain_min)
    end_flow = step7_end_flow_events(beginning_rain_checked, end_rain_checked, beginning_core, end_core, flow, fluct_rain_Tr,
                          fluct_flow_Tr, Tr)
    beginning_flow = step8_beginning_flow_events(beginning_rain_checked, end_rain_checked, end_flow, beginning_core, fluct_rain_Tr,
                                fluct_flow_Tr)
    beginning_rain_ungrouped, end_rain_ungrouped, beginning_flow_ungrouped, end_flow_ungrouped = \
        step9_checks_on_flow_events(beginning_rain_checked, end_rain_checked, beginning_flow, end_flow, fluct_flow_Tr)
    BEGINNING_RAIN, END_RAIN, BEGINNING_FLOW, END_FLOW = step10_check_overlapping_events(beginning_rain, end_rain, beginning_flow, end_flow, time)
    print(beginning_rain_ungrouped, end_rain_ungrouped, beginning_flow_ungrouped, end_flow_ungrouped)
    print(BEGINNING_RAIN, END_RAIN, BEGINNING_FLOW, END_FLOW)