from datetime import timedelta, datetime
import pandas as pd
from obspy.clients.fdsn import Client as FDSN_Client

client = FDSN_Client("GEONET")
start = datetime(2000, 12, 31)
end = datetime(2000, 12, 31)
step = timedelta(days=50)


def save_events(cat):
    event_ids = []
    event_times = []
    latitudes = []
    longitudes = []
    magnitudes = []
    depths = []
    for i in range(len(cat)):
        earthquakeEvent = cat[i]
        event_id = earthquakeEvent.resource_id.id
        event_time = earthquakeEvent.preferred_origin().time
        latitude = earthquakeEvent.preferred_origin().latitude
        longitude = earthquakeEvent.preferred_origin().longitude
        magnitude = round(earthquakeEvent.preferred_magnitude().mag, 1)
        depth = round(earthquakeEvent.preferred_origin().depth / 1000)
        event_ids.append(event_id)
        event_times.append(event_time)
        latitudes.append(latitude)
        longitudes.append(longitude)
        magnitudes.append(magnitude)
        depths.append(depth)
    data_map = {'event_id': event_ids, 'time': event_times, 'latitude': latitudes, 'longitude': longitudes,
                'magnitude': magnitudes, 'depth': depths}
    df = pd.DataFrame(data=data_map)
    df.to_pickle('../datasets/sets/events_temp.pkl')


def download_events():
    cat = None
    cur = start
    while cur < end:
        next_time = cur + step
        if next_time > end:
            next_time = end
        cur_cat = client.get_events(starttime=cur, endtime=next_time, minlatitude=-47.749, maxlatitude=-33.779,
                                    minlongitude=166.104, maxlongitude=178.990)
        if cat is None:
            cat = cur_cat
        else:
            cat.extend(cur_cat)
        cur = next_time
    save_events(cat)


def download_stations():
    inventory = client.get_stations(starttime=start, endtime=end, minlatitude=-47.749, maxlatitude=-33.779,
                                    minlongitude=166.104, maxlongitude=178.990, level="response", location="10",
                                    channel="HHZ")
    stations = []
    longitudes = []
    latitudes = []
    sites = []
    for station in inventory[0]:
        stations.append(station.code)
        latitudes.append(station.latitude)
        longitudes.append(station.longitude)
        sites.append(station.site.name)
    data_map = {'station_code': stations, 'longitude': longitudes, 'latitude': latitudes, 'site': sites}
    df = pd.DataFrame(data=data_map)
    df.to_pickle('../datasets/sets/stations_temp.pkl')
