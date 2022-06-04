import mpl_toolkits
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import numpy as np
import pickle
import os
import logging
from obspy import read
from obspy.clients.fdsn.mass_downloader import RectangularDomain, Restrictions, MassDownloader
from scipy import signal
from obspy import UTCDateTime
from obspy.clients.fdsn import Client as FDSN_Client
from obspy import read_inventory
import asyncio
from itertools import islice

events_df = pd.read_pickle('data/events_processed.pkl')
stations_df = pd.read_pickle('data/stations_processed.pkl')

#events_full = events_df
#events = events_full[0:10]
events = events_df

mdl = MassDownloader(providers=['GEONET'])
def mass_data_downloader(start, stop, event_id, Station,
                         Network='NZ', 
                         Channel='HHZ', 
                         Location=10
                         ):
    '''
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
    '''

    domain = RectangularDomain(
        minlatitude=-47.749,
        maxlatitude=-33.779,
        minlongitude=166.104,
        maxlongitude=178.990
    )

    restrictions = Restrictions(
        starttime = start,
        endtime = stop,
        chunklength_in_sec = None,
        network = Network,
        station = Station,
        location = Location,
        channel = Channel,
        reject_channels_with_gaps = False,
        minimum_length = 0.0,
        minimum_interstation_distance_in_m = 100.0
    )

    #mdl = MassDownloader(providers=['GEONET'])
    ev_str = str(event_id).replace(":", "_")
    mdl.download(
        domain,
        restrictions,
        mseed_storage=f"datasets/normal/waveforms/{ev_str}",
        stationxml_storage="datasets/normal/stations",
    )
    print('done: ',event_id)

logger = logging.getLogger("obspy.clients.fdsn.mass_downloader")
logger.setLevel(logging.WARNING)

async def final_download_threaded(events):
    tasks = []
    print("Initiating mass download request.")
    for i, event in events.iterrows():
        event_id = event.event_id
        event_time = event['time']  
        start=event_time - 30
        end=event_time + 30
        stations = ",".join([station.station_code for j, station in stations_df.iterrows()])
        tasks.append(asyncio.to_thread(mass_data_downloader, start, end, event_id, stations))
        
    await asyncio.gather(*tasks)

#VARIABLES FOR DOWNLOAD
T_event = 30
H_event = 30
threads_at_once = 100
##
for event_sublist in [events[x:x+threads_at_once] for x in range(0, len(events), threads_at_once)]:
    await final_download_threaded(event_sublist)