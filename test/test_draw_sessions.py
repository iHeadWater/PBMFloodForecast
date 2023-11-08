

def test_plot_sessions():
    '''
        wjy_calibrate_time = pd.read_excel(os.path.join(definitions.ROOT_DIR, 'example/洪水率定时间.xlsx'))
        wjy_calibrate_time['starttime'] = pd.to_datetime(wjy_calibrate_time['starttime'], format='%Y/%m/%d %H:%M:%S')
        wjy_calibrate_time['endtime'] = pd.to_datetime(wjy_calibrate_time['endtime'], format='%Y/%m/%d %H:%M:%S')
        for i in range(14, 25):
            start_time = wjy_calibrate_time['starttime'][i]
            end_time = wjy_calibrate_time['endtime'][i]
            x = pd.date_range(start_time, end_time, freq='H')
            fig, ax = plt.subplots(figsize=(9, 6))
            p = ax.twinx()
            filtered_rain_aver_df.index = pd.to_datetime(filtered_rain_aver_df.index)
            flow_mm_h.index = pd.to_datetime(flow_mm_h.index)
            y_rain = filtered_rain_aver_df[start_time: end_time]
            y_flow = flow_mm_h[start_time:end_time]
            ax.bar(x, y_rain.to_numpy().flatten(), color='red', edgecolor='k', alpha=0.6, width=0.04)
            ax.set_ylabel('rain(mm)')
            ax.invert_yaxis()
            p.plot(x, y_flow, color='green', linewidth=2)
            p.set_ylabel('flow(mm/h)')
            plt.savefig(os.path.join(definitions.ROOT_DIR, 'example/rain_flow_event_'+str(start_time).split(' ')[0]+'_wy.png'))
        # XXX_FLOW 和 XXX_RAIN 长度不同，原因暂时未知，可能是数据本身问题（如插值导致）或者单位未修整
        plt.figure()
        x = time
        rain_event_array = np.zeros(shape=len(time))
        flow_event_array = np.zeros(shape=len(time))
        for i in range(0, len(BEGINNING_RAIN)):
            rain_event = filtered_rain_aver_df['rain'][BEGINNING_RAIN[i]: END_RAIN[i]]
            beginning_index = np.argwhere(time == BEGINNING_RAIN[i])[0][0]
            end_index = np.argwhere(time == END_RAIN[i])[0][0]
            rain_event_array[beginning_index: end_index + 1] = rain_event
        for i in range(0, len(BEGINNING_FLOW)):
            flow_event = flow_mm_h[BEGINNING_FLOW[i]: END_FLOW[i]]
            beginning_index = np.argwhere(time == BEGINNING_FLOW[i])[0][0]
            end_index = np.argwhere(time == END_FLOW[i])[0][0]
            flow_event_array[beginning_index: end_index + 1] = flow_event
        y_rain = rain_event_array
        y_flow = flow_event_array
        fig, ax = plt.subplots(figsize=(16, 12))
        p = ax.twinx()
        ax.bar(x, y_rain, color='red', alpha=0.6)
        ax.set_ylabel('rain(mm)')
        ax.invert_yaxis()
        p.plot(x, y_flow, color='green', linewidth=2)
        p.set_ylabel('flow(mm/h)')
        plt.savefig(os.path.join(definitions.ROOT_DIR, 'example/rain_flow_events.png'))
        '''