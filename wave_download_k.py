import asyncio
import logging
import os

import numpy as np
import pandas as pd
from obspy import read
from obspy.clients.fdsn.mass_downloader import RectangularDomain, Restrictions, MassDownloader
from collections import ChainMap
import collections
#selected_stations = ['BFZ', 'BKZ', 'DCZ', 'DSZ', 'EAZ', 'HIZ', 'JCZ', 'KHZ', 'KNZ', 'KUZ', 'LBZ', 'LTZ', 'MLZ',
#                         'MQZ', 'MRZ', 'MSZ', 'MWZ', 'MXZ', 'NNZ', 'ODZ', 'OPRZ', 'OUZ', 'PUZ', 'PXZ', 'QRZ', 'RPZ',
#                         'SYZ', 'THZ', 'TOZ', 'TSZ', 'TUZ', 'URZ', 'VRZ', 'WCZ', 'WHZ', 'WIZ', 'WKZ', 'WVZ']
#selected_stations = ['KHZ', 'MRZ', 'PXZ', 'OPRZ']

events_df = None

def mass_data_downloader(start, stop, event_id, Station, Network='NZ', Channel='HHZ', Location=10):
    """
    This function uses the FDSN mass data downloader to automatically download
    data from the XH network deployed on the RIS from Nov 2014 - Nov 2016.
    More information on the Obspy mass downloader available at:
    https://docs.obspy.org/packages/autogen/obspy.clients.fdsn.mass_downloader.html
    Inputs:
    start: "YYYYMMDD"
    stop:  "YYYYMMDD"
    Network: 2-character FDSN network code
    Station: 2-character station code
    Channel: 3-character channel code
    Location: 10.
    """
    domain = RectangularDomain(
        minlatitude=-47.749,
        maxlatitude=-33.779,
        minlongitude=166.104,
        maxlongitude=178.990
    )
    restrictions = Restrictions(
        starttime=start,
        endtime=stop,
        chunklength_in_sec=None,
        network=Network,
        station=Station,
        location=Location,
        channel=Channel,
        reject_channels_with_gaps=False,
        minimum_length=0.0,
        minimum_interstation_distance_in_m=100.0
    )
    ev_str = str(event_id).replace(":", "_")
    try:
        mdl.download(domain, restrictions,
                     mseed_storage=f"./datasets/{folder}/waveforms/{ev_str}",
                     stationxml_storage=f"./datasets/{folder}/stations")
    except Exception as e:
        print(f'Event: {ev_str}. Error: {e}')
        pass

async def final_download_threaded(events, T, H):
    tasks = []
    for i, event in events.iterrows():
        event_id = event.event_id
        event_time = event['time']
        start = event_time - T - H
        end = event_time - H
        stations = ",".join([station.station_code for j, station in stations_df.iterrows()])
        tasks.append(asyncio.to_thread(mass_data_downloader, start, end, event_id, stations))
    await asyncio.gather(*tasks)

async def final_download():
    print(f'Downloading {folder} waves')
    counter = threads_at_once
    for event_sublist in [events_df[x:x + threads_at_once] for x in range(0, len(events_df), threads_at_once)]:
        print(f'Current batch: {counter}/{len(events_df)}')
        counter += threads_at_once
        await final_download_threaded(event_sublist, T_event, H_event)

def open_file(cur_dir, station, event):
    station_name = station.split('.')[1]
    try:
        file_name = cur_dir + '/' + station
        station_data = np.array(read(file_name)[0].data)
    except Exception as e:
        print(f'Event: {event}, Station: {station_name}. Error: {e}')
        return {}
    if min(station_data) == -14822981 or max(station_data) == -14822981:
        print('Corrupted data')
        return {}

    return {station_name: station_data}

def closest_station_for_data(x: str, events_df) -> str:
    try:
        row = events_df.loc[x]
    except:
        return ''
    return row['closest_station']
def filter_dirs(x:str, y:str, selected_stations) -> bool:
    if y == '':
        return False
    return y in selected_stations
async def process_waves(events_df, selected_stations: list[str], folder: str, closest_only: bool, limit_events=1000000000):
    print(f'Processing {folder} waves')
    path = f'./datasets/{folder}/waveforms/smi_nz.org.geonet/'
    events = os.listdir(path)
    if folder == "active" and closest_only:
        e_df = events_df
        e_df['event_id'] = events_df['event_id'].apply(lambda x: x.split("/")[1])
        e_df = e_df.set_index('event_id')
        closest_stations = list(map(lambda x: closest_station_for_data(x, e_df), events))
        print(dict(collections.Counter(closest_stations)))
        events = list(filter(lambda x: filter_dirs(x[1], closest_stations[x[0]], selected_stations), enumerate(events)))
        events = [e for i, e in events]
    events = events[:limit_events]
    print(len(events))
    final_data = {}
    for j, event in enumerate(events):
        if j % 100 == 0:
            print(f'Current batch: {j}/{len(events)}')
        cur_dir = path + event
        station_files = os.listdir(cur_dir)
        station_files = list(filter(lambda x: x.split(".")[1] in selected_stations, station_files))
        if len(station_files) < len(selected_stations):
            continue
        station_data_arr = {}
        tasks = []
        for station in station_files:
            tasks.append(asyncio.to_thread(open_file, cur_dir, station, event))
        res = await asyncio.gather(*tasks)
        station_data_arr = dict(ChainMap(*res))
        if len(station_data_arr) >= len(selected_stations):
            final_data[event] = station_data_arr
    final_data = pd.DataFrame(final_data).transpose()
    final_data.to_pickle(f'./datasets/{folder}/waves_full.pkl')
    return final_data

#mdl = MassDownloader(providers=['GEONET'])
mdl = None
#logger = logging.getLogger("obspy.clients.fdsn.mass_downloader")
#logger.setLevel(logging.WARNING)

threads_at_once = 100

# Parameters
folder = "normal"

#print(events_df['closest_station'].value_counts())
stations_df = None
H_event = 0
#if folder == "active":
#    events_df = pd.read_pickle('datasets/sets/events_processed.pkl')
#    H_event = 0
#else:
#    events_df = pd.read_pickle('datasets/sets/events_normal.pkl')
#    H_event = 2000
#stations_df = pd.read_pickle('datasets/sets/stations_processed.pkl')

# TODO Active 2000
# TODO Normal 2000
# TODO T_event 30
T_event = 30
#events_df = events_df[500:2000]

#asyncio.run(final_download())
def preprocess(selected_stations: list[str], closest_only: bool):
    events_df = pd.read_pickle('data/events_processed.pkl')
    active_df = asyncio.run(process_waves(events_df, selected_stations, "active", closest_only))
    print(active_df.shape[0])
    asyncio.run(process_waves(events_df, selected_stations, "normal", closest_only, limit_events=round(active_df.shape[0]*1.2)))

if __name__ == "__main__":
    preprocess(['KHZ', 'MRZ', 'PXZ', 'OPRZ'], True)
