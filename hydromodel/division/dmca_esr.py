import numpy as np


def movmean(X, n):
    ones = np.ones(X.shape)
    kernel = np.ones(n)
    return np.convolve(X, kernel, mode='same') / np.convolve(ones, kernel, mode='same')


def STEP1_STEP2_Tr_and_fluctuations_timeseries(rain, flow, rain_min, max_window):
    rain = rain.T
    flow = flow.T
    rain_int = np.nancumsum(rain)
    flow_int = np.nancumsum(flow)
    T = rain.size
    rain_mean = np.empty(((max_window - 1) // 2, T))
    flow_mean = np.empty(((max_window - 1) // 2, T))
    fluct_rain = np.empty(((max_window - 1) // 2, T))
    fluct_flow = np.empty(((max_window - 1) // 2, T))
    F_rain = np.empty((max_window - 1) // 2)
    F_flow = np.empty((max_window - 1) // 2)
    F_rain_flow = np.empty((max_window - 1) // 2)
    rho = np.empty((max_window - 1) // 2)
    for window in np.arange(3, max_window + 1, 2):
        int_index = int((window - 1) / 2 - 1)
        start_slice = int(window - 0.5 * (window - 1))
        dst_slice = int(T - 0.5 * (window - 1))
        # 新建一个循环体长度*数据长度的大数组
        rain_mean[int_index] = movmean(rain_int, window)
        flow_mean[int_index] = movmean(flow_int, window)
        fluct_rain[int_index] = rain_int - rain_mean[int_index, :]
        F_rain[int_index] = (1 / (T - window + 1)) * np.nansum(
            (fluct_rain[int_index, start_slice:dst_slice]) ** 2)
        fluct_flow[int_index, np.newaxis] = flow_int - flow_mean[int_index, :]
        F_flow[int_index] = (1 / (T - window + 1)) * np.nansum(
            (fluct_flow[int_index, start_slice:dst_slice]) ** 2)
        F_rain_flow[int_index] = (1 / (T - window + 1)) * np.nansum(
            (fluct_rain[int_index, start_slice:dst_slice]) * (
                fluct_flow[int_index, start_slice:dst_slice]))
        rho[int_index] = F_rain_flow[int_index] / (
                np.sqrt(F_rain[int_index]) * np.sqrt(F_flow[int_index]))
    pos_min = np.argmin(rho)
    Tr = pos_min + 1
    tol_fluct_rain = (rain_min / (2 * Tr + 1)) * Tr
    tol_fluct_flow = flow_int[-1] / 1e15
    fluct_rain[pos_min, np.fabs(fluct_rain[pos_min, :]) < tol_fluct_rain] = 0
    fluct_flow[pos_min, np.fabs(fluct_flow[pos_min, :]) < tol_fluct_flow] = 0
    fluct_rain_Tr = fluct_rain[pos_min, :]
    fluct_flow_Tr = fluct_flow[pos_min, :]
    fluct_bivariate_Tr = fluct_rain_Tr * fluct_flow_Tr
    fluct_bivariate_Tr[np.fabs(fluct_bivariate_Tr) < np.finfo(np.float64).eps] = 0  # 便于比较
    return Tr, fluct_rain_Tr, fluct_flow_Tr, fluct_bivariate_Tr


def STEP3_core_identification(fluct_bivariate_Tr):
    d = np.diff(fluct_bivariate_Tr, prepend=[0], append=[0])  # 计算相邻数值差分，为0代表两端点处于0区间
    d[np.fabs(d) < np.finfo(np.float64).eps] = 0  # 确保计算正确
    d = np.logical_not(d)  # 求0-1数组，为真代表为0区间
    d0 = np.logical_not(np.convolve(d, [1, 1], 'valid'))  # 对相邻元素做OR，代表原数组数值是否处于某一0区间，再取反表示取有效值
    valid = np.logical_or(fluct_bivariate_Tr, d0)  # 有效core
    d_ = np.diff(valid, prepend=[0], append=[0])  # 求差分方便取上下边沿
    beginning_core = np.argwhere(d_ == 1)  # 上边沿为begin
    end_core = np.argwhere(d_ == -1) - 1  # 下边沿为end
    return beginning_core, end_core


def STEP4_end_rain_events(beginning_core, end_core, rain, fluct_rain_Tr, rain_min):
    end_rain = end_core.copy()
    rain = rain.T
    for g in range(end_core.size):
        if fluct_rain_Tr[end_core[g]] < np.finfo(np.float64).eps and (
                end_core[g] + 1 >= fluct_rain_Tr.size or fluct_rain_Tr[end_core[g] + 1] < np.finfo(np.float64).eps):
            # case 1&2
            if rain[end_core[g]] < np.finfo(np.float64).eps:
                # case 1
                while end_rain[g] > beginning_core[g] and rain[end_rain[g]] < np.finfo(np.float64).eps:
                    end_rain[g] = end_rain[g] - 1
            else:
                # case 2
                bound = beginning_core[g + 1] if g + 1 < beginning_core.size else rain.size
                while end_rain[g] < bound and rain[end_rain[g]] > rain_min:
                    end_rain[g] = end_rain[g] + 1
                end_rain[g] = end_rain[g] - 1  # 回到最后一个
        else:
            # case 3
            # 若在降水，先跳过
            while end_rain[g] >= beginning_core[g] and rain[end_rain[g]] > rain_min:
                end_rain[g] = end_rain[g] - 1
            while end_rain[g] >= beginning_core[g] and rain[end_rain[g]] < rain_min:
                end_rain[g] = end_rain[g] - 1
    return end_rain


def STEP5_beginning_rain_events(beginning_core, end_rain, rain, fluct_rain_Tr, rain_min):
    beginning_rain = beginning_core.copy()
    rain = rain.T
    for g in range(beginning_core.size):
        if fluct_rain_Tr[beginning_core[g]] < np.finfo(np.float64).eps \
                and (beginning_core[g] + 1 >= fluct_rain_Tr.size or fluct_rain_Tr[beginning_core[g] + 1] < np.finfo(np.float64).eps) \
                and rain[beginning_core[g]] < np.finfo(np.float64).eps:
            # case 1
            while beginning_rain[g] < end_rain[g] and rain[beginning_rain[g]] < np.finfo(np.float64).eps:
                beginning_rain[g] = beginning_rain[g] + 1
        else:
            # case 2&3
            bound = end_rain[g - 1] if g - 1 > 0 else -1
            while beginning_rain[g] > bound and rain[beginning_rain[g]] > rain_min:
                beginning_rain[g] = beginning_rain[g] - 1
            end_rain[g] = end_rain[g] + 1  # 回到第一个
    return beginning_rain


# very bad, ignoring it
def STEP6_checks_on_rain_events(beginning_rain, end_rain, rain, rain_min):
    return (beginning_rain, end_rain)
