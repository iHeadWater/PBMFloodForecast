import numpy as np
from scipy.signal import convolve


def step1_2_tr_and_fluctuations_timeseries(rain, flow, rain_min, max_window):
    rain = rain.ravel()  # convert to 1D array
    flow = flow.ravel()
    rain_int = np.nancumsum(rain)  # cumulative sum
    flow_int = np.nancumsum(flow)
    T = len(rain)
    rain_mean = []
    flow_mean = []
    F_rain = []
    F_flow = []
    F_rain_flow = []
    rho = []
    fluct_rain = np.zeros((int((max_window - 2)/2), len(rain_int)))
    fluct_flow = np.zeros((int((max_window - 2)/2), len(flow_int)))
    for window in range(3, max_window + 1, 2):
        kernel = np.ones(window) / window
        rain_mean.append(convolve(rain_int, kernel, mode='same'))
        flow_mean.append(convolve(flow_int, kernel, mode='same'))
        # fluct_flow和fluct_rain是二维数组，shape = (window_amount * len(flow_int))
        fluct_rain[int(window/2)-1] = rain_int - rain_mean[-1]
        F_rain.append(np.nansum(fluct_rain[:, window - 1:] ** 2) / (T - window + 1))
        fluct_flow[int(window/2)-1] = flow_int - flow_mean[-1]
        F_flow.append(np.nansum(fluct_flow[:, window - 1:] ** 2) / (T - window + 1))
        F_rain_flow.append(np.nansum(fluct_rain[:, window - 1:] * fluct_flow[:, window - 1:]) / (T - window + 1))
        rho.append(F_rain_flow[-1] / np.sqrt(F_rain[-1] * F_flow[-1]))
    Tr = np.nanargmin(rho)
    tol_fluct_rain = (rain_min / (2 * Tr + 1)) * (((2 * Tr + 1) - 1) / 2)
    tol_fluct_flow = flow_int[-1] / 1e15
    # fluct_rain和fluct_flow是二维数组所以要取Tr而非[:Tr]
    fluct_rain_Tr = fluct_rain[Tr].copy()
    fluct_rain_Tr[np.abs(fluct_rain_Tr) < tol_fluct_rain] = 0
    fluct_flow_Tr = fluct_flow[Tr].copy()
    fluct_flow_Tr[np.abs(fluct_flow_Tr) < tol_fluct_flow] = 0
    fluct_bivariate_Tr = fluct_rain_Tr * fluct_flow_Tr
    return Tr, fluct_rain_Tr, fluct_flow_Tr, fluct_bivariate_Tr


def step3_core_identification(fluct_bivariate_Tr):
    beginning_core = []
    end_core = []
    g = 1
    q = 1
    while q + 1 <= len(fluct_bivariate_Tr):
        if np.abs(fluct_bivariate_Tr[q]) > 0:
            beginning_core.append(q)
            while q + 1 < len(fluct_bivariate_Tr) and np.sum(np.abs(fluct_bivariate_Tr[q:q + 2])) > 0:
                q += 1
                if q >= len(fluct_bivariate_Tr):
                    break
            end_core.append(q - 1)
            g += 1
        q += 1
    beginning_core = np.array(beginning_core)
    end_core = np.array(end_core)
    return beginning_core, end_core


def step4_end_rain_events(beginning_core, end_core, rain, fluct_rain_Tr, rain_min):
    rain = rain.ravel()  # flatten
    end_rain = np.zeros(len(end_core))
    for g in range(len(end_core)):
        # Preliminary guess
        end_rain[g] = end_core[g]
        # Core ends because rain fluctuations are zero (case 1 and 2)
        if end_core[g] + 2 < len(rain) and np.sum(fluct_rain_Tr[end_core[g] + 1: end_core[g] + 2] == 0) == 2:
            # Case 1: Core ends, rain is zero
            if rain[int(end_rain[g])] == 0:
                while end_rain[g] - 1 > 0 and rain[int(end_rain[g])] == 0:
                    end_rain[g] -= 1
            # Case 2: Core ends, rain is non-zero
            else:
                next_core = np.where(np.isnan(beginning_core[g + 1:]) is False)[0]
                if len(next_core) > 0:
                    next_core = beginning_core[g + 1:][next_core[0]]
                    while (end_rain[g] + 1 < len(rain) and
                           rain[int(end_rain[g])] > rain_min and
                           end_rain[g] < next_core):
                        end_rain[g] += 1
                    end_rain[g] -= 1

                else:  # Last event
                    while (end_rain[g] + 1 < len(rain) and
                           rain[int(end_rain[g])] > rain_min):
                        end_rain[g] += 1
                    end_rain[g] -= 1
        # Case 3: Core ends because flow fluctuations are zero
        else:
            while (end_rain[g] - 1 > 0 and
                   rain[int(end_rain[g])] > rain_min and
                   end_rain[g] >= beginning_core[g]):
                end_rain[g] -= 1

            while (end_rain[g] - 1 > 0 and
                   rain[int(end_rain[g])] < rain_min and
                   end_rain[g] >= beginning_core[g]):
                end_rain[g] -= 1
    return end_rain


def step5_beginning_rain_events(beginning_core, end_rain, rain, fluct_rain_Tr, rain_min):
    rain = rain.ravel()
    beginning_rain = np.zeros(len(beginning_core))
    for g in range(len(beginning_core)):
        # Preliminary guess
        beginning_rain[g] = beginning_core[g]
        # Before core, rain fluctuations are zero (case 1 and 2)
        if beginning_core[g] > 2 and np.sum(fluct_rain_Tr[beginning_core[g] - 2: beginning_core[g] - 1] == 0) == 2:
            # Case 1: Core starts, rain is zero
            if rain[int(beginning_core[g])] == 0:
                while beginning_rain[g] + 1 < len(rain) and rain[int(beginning_rain[g])] == 0:
                    beginning_rain[g] += 1

            # Case 2: Core starts, rain is non-zero
            else:
                prev_end = np.where(np.isnan(end_rain[:g]) is False)[0]
                if len(prev_end) > 0:
                    prev_end = end_rain[prev_end[-1]]
                    while (beginning_rain[g] - 1 > 0 and
                           rain[int(beginning_rain[g])] > rain_min and
                           beginning_rain[g] > prev_end):
                        beginning_rain[g] -= 1
                    beginning_rain[g] += 1

                else:  # First event
                    while beginning_rain[g] - 1 > 0 and rain[int(beginning_rain[g])] > rain_min:
                        beginning_rain[g] -= 1
                    beginning_rain[g] += 1
        # Before core, flow fluctuations are zero (case 3)
        else:
            prev_end = np.where(np.isnan(end_rain[:g]) is False)[0]
            if len(prev_end) > 0:
                prev_end = end_rain[prev_end[-1]]
                while (beginning_rain[g] - 1 > 0 and
                       rain[int(beginning_rain[g])] > rain_min and
                       beginning_rain[g] > prev_end):
                    beginning_rain[g] -= 1
                beginning_rain[g] += 1
            else:  # First event
                while beginning_rain[g] - 1 > 0 and rain[int(beginning_rain[g])] > rain_min:
                    beginning_rain[g] -= 1
                beginning_rain[g] += 1
    return beginning_rain


def step6_check_rain_events(beginning_rain, end_rain, rain, rain_min):
    rain = rain.ravel()
    beginning_rain_checked = np.zeros(len(beginning_rain))
    beginning_rain_checked[:] = np.nan
    end_rain_checked = np.zeros(len(end_rain))
    end_rain_checked[:] = np.nan
    for g in range(len(beginning_rain)):
        if (beginning_rain[g] > end_rain[g] or
                rain[int(beginning_rain[g] - 1)] > rain_min or
                rain[int(end_rain[g] + 1)] > rain_min):
            pass
        else:
            beginning_rain_checked[g] = beginning_rain[g]
            end_rain_checked[g] = end_rain[g]
    return beginning_rain_checked, end_rain_checked


def step7_end_flow_events(beginning_rain_checked, end_rain_checked, beginning_core, end_core, flow, fluct_rain_Tr,
                          fluct_flow_Tr, Tr):
    end_flow = np.full(len(end_rain_checked), np.nan)
    for g in range(len(end_rain_checked)):
        if not np.isnan(end_rain_checked[g]) and not np.isnan(beginning_rain_checked[g]):
            if end_core[g] + 2 < len(flow) and sum(fluct_rain_Tr[end_core[g] + 1:end_core[g] + 2] == 0) == 2:
                # Preliminary guess
                end_flow[g] = end_rain_checked[g]
                next_event_beginning = np.nonzero(np.isnan(beginning_rain_checked[g + 1:]) == 0)[0]
                if len(next_event_beginning) > 0:
                    # Move forward until end of negative fluct_flow_Tr
                    while end_flow[g] + 1 < len(flow) and fluct_flow_Tr[end_flow[g]] <= 0 and end_flow[g] < \
                            beginning_rain_checked[g + next_event_beginning[0]] + Tr:
                        end_flow[g] += 1
                    # Move forward until end of positive fluct_flow_Tr
                    while end_flow[g] + 1 < len(flow) and fluct_flow_Tr[end_flow[g]] > 0 and end_flow[g] < \
                            beginning_rain_checked[g + next_event_beginning[0]] + Tr:
                        end_flow[g] += 1
                    end_flow[g] -= 1
                else:
                    # Move forward until end of negative fluct_flow_Tr
                    while end_flow[g] + 1 < len(flow) and fluct_flow_Tr[end_flow[g]] <= 0:
                        end_flow[g] += 1
                    # Move forward until end of positive fluct_flow_Tr
                    while end_flow[g] + 1 < len(flow) and fluct_flow_Tr[end_flow[g]] > 0:
                        end_flow[g] += 1
                    end_flow[g] -= 1
            else:
                # Preliminary guess
                end_flow[g] = end_core[g]
                # 索引问题，加个int
                while end_flow[g] > beginning_core[g] and fluct_flow_Tr[int(end_flow[g])] <= 0:
                    end_flow[g] -= 1
        else:
            end_flow[g] = np.nan
            # 这里因为matlab和python语法不同注释掉一行
            # beginning_flow[g] = np.nan
    return end_flow


def step8_beginning_flow_events(beginning_rain_checked, end_rain_checked, end_flow, beginning_core, fluct_rain_Tr,
                                fluct_flow_Tr):
    beginning_flow = np.full(len(beginning_rain_checked), np.nan)
    for g in range(len(beginning_rain_checked)):
        if not np.isnan(beginning_rain_checked[g]) and not np.isnan(end_rain_checked[g]):
            if beginning_core[g] > 2 and sum(fluct_rain_Tr[beginning_core[g] - 2:beginning_core[g] - 1] == 0) == 2:
                # Preliminary guess
                beginning_flow[g] = beginning_rain_checked[g]
                while fluct_flow_Tr[beginning_flow[g]] > 0 and beginning_flow[g] < end_flow[g]:
                    beginning_flow[g] += 1
            else:
                # Preliminary guess
                beginning_flow[g] = beginning_core[g]
                # 索引问题，加个int
                while beginning_flow[g] <= end_flow[g] and fluct_flow_Tr[int(beginning_flow[g])] >= 0:
                    beginning_flow[g] += 1
    return beginning_flow


def step9_checks_on_flow_events(beginning_rain_checked, end_rain_checked, beginning_flow, end_flow, fluct_flow_Tr):
    beginning_flow_checked = np.full(len(beginning_flow), np.nan)
    end_flow_checked = np.full(len(end_flow), np.nan)
    for g in range(len(beginning_flow)):
        # 索引问题，加个int
        if not np.isnan(beginning_flow[g]) and not np.isnan(end_flow[g]) and (
                end_flow[g] <= beginning_flow[g] or fluct_flow_Tr[int(beginning_flow[g])] > 0 or fluct_flow_Tr[int(end_flow[g])]
                < 0 or beginning_flow[g] < beginning_rain_checked[g] or end_flow[g] < end_rain_checked[g]):
            beginning_flow_checked[g] = np.nan
            end_flow_checked[g] = np.nan
        else:
            beginning_flow_checked[g] = beginning_flow[g]
            end_flow_checked[g] = end_flow[g]
    # Select only events with valid rain and flow
    valid_events = np.nonzero((np.isnan(beginning_rain_checked) == 0) & (np.isnan(beginning_flow_checked) == 0) & (np.isnan(
        end_rain_checked) == 0) & (np.isnan(end_flow_checked) == 0))[0]
    beginning_flow_ungrouped = beginning_flow_checked[valid_events]
    end_flow_ungrouped = end_flow_checked[valid_events]
    beginning_rain_ungrouped = beginning_rain_checked[valid_events]
    end_rain_ungrouped = end_rain_checked[valid_events]
    return beginning_rain_ungrouped, end_rain_ungrouped, beginning_flow_ungrouped, end_flow_ungrouped


def step10_check_overlapping_events(beginning_rain, end_rain, beginning_flow, end_flow, time):
    marker_overlapping = []
    for g in range(len(end_rain) - 1):
        if (end_rain[g] > beginning_rain[g + 1] or
                end_flow[g] > beginning_flow[g + 1]):
            marker_overlapping.append(g)
    if marker_overlapping:
        for q in range(len(marker_overlapping)):
            to_group = [marker_overlapping[q]]
            while (q < len(marker_overlapping) - 1 and
                   marker_overlapping[q + 1] == marker_overlapping[q] + 1):
                to_group.append(marker_overlapping[q + 1])
                q += 1
            beginning_rain[to_group[0]] = beginning_rain[to_group[0]]
            beginning_flow[to_group[0]] = beginning_flow[to_group[0]]
            end_flow[to_group[0]] = end_flow[to_group[-1] + 1]
            end_rain[to_group[0]] = end_rain[to_group[-1] + 1]
            if len(to_group) > 1:
                beginning_rain[to_group[1:]] = np.nan
                beginning_flow[to_group[1:]] = np.nan
                end_flow[to_group[1:]] = np.nan
                end_rain[to_group[1:]] = np.nan
            beginning_rain[to_group[-1] + 1] = np.nan
            beginning_flow[to_group[-1] + 1] = np.nan
            end_flow[to_group[-1] + 1] = np.nan
            end_rain[to_group[-1] + 1] = np.nan
    valid = ~np.isnan(beginning_rain) & ~np.isnan(beginning_flow) & ~np.isnan(end_rain) & ~np.isnan(end_flow)
    beginning_flow = beginning_flow[valid]
    end_flow = end_flow[valid]
    beginning_rain = beginning_rain[valid]
    end_rain = end_rain[valid]
    BEGINNING_RAIN = time[beginning_rain.astype(int)]
    END_RAIN = time[end_rain.astype(int)]
    BEGINNING_FLOW = time[beginning_flow.astype(int)]
    END_FLOW = time[end_flow.astype(int)]
    return BEGINNING_RAIN, END_RAIN, BEGINNING_FLOW, END_FLOW
