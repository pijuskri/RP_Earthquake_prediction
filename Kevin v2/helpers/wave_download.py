import asyncio
import logging
import pandas as pd
from obspy.clients.fdsn.mass_downloader import RectangularDomain, Restrictions, MassDownloader

mdl = MassDownloader(providers=['GEONET'])
logger = logging.getLogger("obspy.clients.fdsn.mass_downloader")
logger.setLevel(logging.WARNING)
threads_at_once = 100


def mass_data_downloader(folder, start, stop, event_id, station, Network='NZ', Channel='HHZ', Location=10):
    """
    https://docs.obspy.org/packages/autogen/obspy.clients.fdsn.mass_downloader.html
    Network: 2-character FDSN network code
    Station: 2-character station code
    Channel: 3-character channel code
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
        station=station,
        chunklength_in_sec=None,
        network=Network,
        location=Location,
        channel=Channel,
        reject_channels_with_gaps=False,
        minimum_length=0.0,
        minimum_interstation_distance_in_m=100.0
    )
    ev_str = str(event_id).replace(":", "_").split('/')[-1]
    try:
        mdl.download(domain, restrictions,
                     mseed_storage=f"../datasets/{folder}/waveforms/{ev_str}",
                     stationxml_storage=f"../datasets/{folder}/stations")
    except Exception as e:
        print(f'Event: {ev_str}. Error: {e}')
        pass


async def final_download_threaded(folder, events, stations, T, H):
    tasks = []
    for i, event in events.iterrows():
        event_id = event.event_id
        event_time = event['time']
        start = event_time - T - H
        stop = event_time - H
        station = ",".join([station.station_code for j, station in stations.iterrows()])
        tasks.append(asyncio.to_thread(mass_data_downloader, folder, start, stop, event_id, station))
    await asyncio.gather(*tasks)


async def final_download(folder, start, stop):
    T_event = 60
    stations_df = pd.read_pickle('../datasets/sets/stations.pkl')
    if folder == "active":
        events_df = pd.read_pickle('../datasets/sets/events.pkl')
        H_event = 0
    else:
        events_df = pd.read_pickle('../datasets/sets/normal.pkl')
        H_event = 2000
    events_df = events_df[start:stop]
    print(f'Downloading {folder} waves')
    counter = threads_at_once
    for event_sublist in [events_df[x:x + threads_at_once] for x in range(0, len(events_df), threads_at_once)]:
        print(f'Current batch: {counter}/{len(events_df)}')
        counter += threads_at_once
        await final_download_threaded(folder, event_sublist, stations_df, T_event, H_event)


# Active/Normal 10k
a = 10000
b = 11000
print(f'Downloading from {a} to {b}')
asyncio.run(final_download('active', a, b))
asyncio.run(final_download('normal', a, b))
