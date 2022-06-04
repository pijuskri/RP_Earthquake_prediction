import pandas as pd


def sort_by_time(events):
    events = events.sort_values(by=['time'])
    events.to_pickle('../datasets/sets/events_temp.pkl')


def filter_normal(events):
    new_events = pd.DataFrame({}, columns=events.columns)
    prev_time = None
    for i, event in events.iterrows():
        if i % 1000 == 0:
            print(f'{i}/{events.shape[0]}')
        time = event['time']
        if not prev_time:
            prev_time = time
            continue
        if (time - prev_time) > 5000:
            new_events.loc[len(new_events.index)] = event
        prev_time = time
    new_events.to_pickle('../datasets/sets/events_temp.pkl')


def filter_events(events):
    events = events[events['magnitude'] > 0.5]
    events = events[events['magnitude'] < 2.5]
    events.to_pickle('../datasets/sets/events_temp.pkl')


def filter_station(stations):
    selected_stations = ['BFZ', 'BKZ', 'DCZ', 'DSZ', 'EAZ', 'HIZ', 'JCZ', 'KHZ', 'KNZ', 'KUZ', 'LBZ', 'LTZ', 'MLZ',
                         'MQZ', 'MRZ', 'MSZ', 'MWZ', 'MXZ', 'NNZ', 'ODZ', 'OPRZ', 'OUZ', 'PUZ', 'PXZ', 'QRZ', 'RPZ',
                         'SYZ', 'THZ', 'TOZ', 'TSZ', 'TUZ', 'URZ', 'VRZ', 'WCZ', 'WHZ', 'WIZ', 'WKZ', 'WVZ']
    new_stations = stations.loc[stations.station_code.isin(selected_stations)]
    new_stations.to_pickle('../datasets/sets/stations_temp.pkl')
