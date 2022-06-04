import mpl_toolkits
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import numpy as np
import pickle
import os
import math
import asyncio

import obspy
from obspy import read
from scipy import signal
from obspy.core.trace import Trace
from obspy import UTCDateTime
from obspy.clients.fdsn import Client as FDSN_Client
from obspy import read_inventory
from obspy.geodetics.base import gps2dist_azimuth
from skimage.measure import block_reduce
from scipy import signal
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler

original_signal = 3001
new_signal = 301
reduction_factor = 10
time_arr = np.linspace(0.0, 30.0, 301)
selected_stations = ['BFZ', 'BKZ', 'DCZ', 'DSZ', 'EAZ', 'HIZ', 'JCZ', 'KHZ', 'KNZ', 'KUZ', 'LBZ', 'LTZ', 'MLZ',
                         'MQZ', 'MRZ', 'MSZ', 'MWZ', 'MXZ', 'NNZ', 'ODZ', 'OPRZ', 'OUZ', 'PUZ', 'PXZ', 'QRZ', 'RPZ',
                         'SYZ', 'THZ', 'TOZ', 'TSZ', 'TUZ', 'URZ', 'VRZ', 'WCZ', 'WHZ', 'WIZ', 'WKZ', 'WVZ']
stations_to_get = len(selected_stations) #=38 #56
#single station#
stations_to_get = 1
cur_station = "MRZ" #selected_stations[7]
##############
scaler = MinMaxScaler(feature_range=(-1, 1))#StandardScaler()#MinMaxScaler(feature_range=(-1, 1))#Normalizer()
norm = Normalizer()


def lower_hz(data, reduction_factor):
    return data[::reduction_factor][:new_signal]

# preprocess event seismic data per each station
def normalize_all(data):
    for i, new_data in enumerate(data):
        data[i] = scaler.transform(new_data.transpose()).transpose()


def preprocess_data(data):
    new_data_arr = []
    for i, new_data in enumerate(data):
        min_data = np.min(new_data)
        max_data = np.max(new_data)
        if len(new_data) < original_signal or min_data == -14822981 or max_data == -14822981:
            return None
        # new_data = Trace(np.array(new_data)).filter('lowpass', freq=0.5, corners=2, zerophase=True).data
        new_data = new_data[::reduction_factor]
        new_data = new_data[:new_signal]
        new_data_arr.append(new_data)

    new_data_arr = norm.fit_transform(np.array(new_data_arr))
    scaler.partial_fit(new_data_arr.transpose())
    return np.array(new_data_arr)

def read_file(station_file):
    return read(station_file)[0].data
def filter_dirs(x: str, events_df) -> bool:
    row = events_df[events_df['event_id'] == x]
    if row.shape[0] != 1:
        return False
    return row['closest_station'].iloc[0] == cur_station
async def process(folder, active: bool, single: bool):
    datasets_location = f'datasets/{folder}/waveforms/smi_nz.org.geonet/'
    list_of_directories = os.listdir(datasets_location)
    #list_of_directories = list_of_directories[:4000]
    events_df = pd.read_pickle('data/events_processed.pkl')
    print(events_df['closest_station'].mode())
    if single and active:
        events_df['event_id'] = events_df['event_id'].apply(lambda x: x.split("/")[1])
        list_of_directories = list(filter(lambda x: filter_dirs(x, events_df), list_of_directories))

    list_of_directories = list_of_directories[:3000]
    print(len(list_of_directories))
    master_data_per_event = []

    cur_path = os.path.join(os.getcwd(), datasets_location)

    for id_event, directory in enumerate(list_of_directories):

        # station files for each event
        cur_dir = os.path.join(cur_path, directory)
        station_files = os.listdir(cur_dir)
        if single:
            station_files = filter(lambda x: x.split(".")[1] == cur_station, station_files)
        else:
            station_files = filter(lambda x: x.split(".")[1] in selected_stations, station_files)
        station_files = [os.path.join(cur_dir, station) for station in station_files]

        if len(station_files) < stations_to_get:
            continue

        # station data per event
        n_to_read = min(len(station_files), stations_to_get)
        station_data_arr = [None] * n_to_read  # np.zeros((n_to_read, original_signal))
        tasks = []
        for i in range(0, n_to_read):
            tasks.append(asyncio.to_thread(read_file, station_files[i]))
            # if corrupted data, continue to next station

        station_data_arr = await asyncio.gather(*tasks)
        station_data_processed = preprocess_data(station_data_arr)
        if station_data_processed is None:
            continue
        master_data_per_event.append(station_data_processed)

        if id_event % 500 == 0:
            print(id_event)

        del station_data_arr

        # os.chdir('../')

    return master_data_per_event

if __name__ == '__main__':
    #folder = "active" #active/normal
    "processing active"
    active_data = asyncio.run(process("active", True, True))
    "processing normal"
    normal_data = asyncio.run(process("normal", False, True))

    print(np.max(active_data), np.min(active_data))
    print(np.max(normal_data), np.min(normal_data))

    normalize_all(active_data)
    normalize_all(normal_data)

    save_loc = os.path.join(os.getcwd(), f'datasets/active/waveforms/100hz/')
    pickle.dump(active_data, open(save_loc + "normal_seismic_100hz.pkl", "wb"))

    save_loc = os.path.join(os.getcwd(), f'datasets/normal/waveforms/100hz/')
    pickle.dump(normal_data, open(save_loc + "normal_seismic_100hz.pkl", "wb"))
    print("DONE!")