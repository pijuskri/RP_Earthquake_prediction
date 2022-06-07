import sys
from math import floor
import pandas as pd
from sklearn.preprocessing import Normalizer, StandardScaler


def sanitize(df, frames):
    pd.options.mode.chained_assignment = None
    frames = frames * 100
    print(f'Before: {df.shape}')
    df = df.dropna()
    for i, row in df.iterrows():
        if (row.str.len() < frames).any():
            df = df.drop(i)
            continue
        df.loc[i] = row.apply(lambda x: x[-frames:])
    print(f'After: {df.shape}')
    return df


def combine_data(normal, active, low, high, flat):
    events = pd.read_pickle('data/events_processed.pkl')
    high_events = events[events['magnitude'] > 2.5]['event_id'].apply(lambda x: x.split('/')[1])
    normal['label'] = 0
    active['label'] = active.index
    active['label'] = active['label'].apply(lambda x: 0 if x in list(high_events) else 1)
    active_low = active[active['label'] == 1]
    active_high = active[active['label'] == 0]

    inf = float('inf')
    low_size = inf if low == 0.0 else len(active_low) / low
    high_size = inf if high == 0.0 else len(active_high) / high
    flat_size = inf if flat == 0.0 else len(normal) / flat
    idx = min([low_size, high_size, flat_size])
    print(f'IDX: {idx}, Low events: {len(active_low)}, High events: {len(active_high)}, Normal events: {len(normal)}')
    return pd.concat([active_low[:floor(idx * low)], active_high[:floor(idx * high)], normal[:floor(idx * flat)]])

def combine_all(normal, active, depth_class=None):
    events = pd.read_pickle('data/events_processed.pkl')
    deep = events[events['depth'] > 70000]['event_id'].apply(lambda x: x.split('/')[1])
    normal['label'] = 0
    active['label'] = 1
    if depth_class == 'deep':
        active = active[active['event_id'] in list(deep)]
    elif depth_class == 'shallow':
        active = active[active['event_id'] not in list(deep)]

    same = min(len(normal), len(active))
    return pd.concat([active[:same], normal[:same]])

def combine_deep_shallow(normal, active, shuffle=False):
    events = pd.read_pickle('data/events_processed.pkl')
    #70
    deep_events = events[events['depth'] > 70]['event_id'].apply(lambda x: x.split('/')[1])
    normal['label'] = 0
    active['label'] = active.index
    active['label'] = active['label'].apply(lambda x: 0 if x in list(deep_events) else 1)

    #shallow = active[~active['event_id'].isin(list(deep_events))]
    #deep = active[active['event_id'].isin(list(deep_events))]
    shallow = active[active['label'] == 1]
    deep = active[active['label'] == 0]
    deep['label'] = 1

    if shuffle:
        shallow = shallow.sample(frac=1)
        deep = deep.sample(frac=1)

    print(f"shallow: {shallow.shape[0]}, deep: {deep.shape[0]}")
    same = min(shallow.shape[0], deep.shape[0])
    shallow = shallow[shallow.shape[0]-same:]
    deep = deep[:same]
    print(f"final sizes of {shallow.shape[0]}")

    same = min(len(normal), shallow.shape[0])
    return pd.concat([shallow[:same], normal[:same]]), pd.concat([deep[:same], normal[:same]])

# TODO Check for bugs
def normalize_scale(df):
    temp = df['label'].copy()
    df = df.drop(columns=['label'])
    norm = Normalizer()
    norm.fit(df.values.flatten().tolist())
    df = df.apply(lambda x: x.apply(lambda y: norm.transform(y.reshape(1, -1))[0]))
    #scale = StandardScaler()
    #scale.fit(df.values.flatten().tolist())
    #df = df.apply(lambda x: x.apply(lambda y: scale.transform(y.reshape(1, -1))[0]))

    df = df.applymap(lambda x: x[::2])
    df['label'] = temp
    return df

def process():
    active = sanitize(pd.read_pickle('./datasets/active/waves_full.pkl'), 30)
    normal = sanitize(pd.read_pickle('./datasets/normal/waves_full.pkl'), 30)
    #dataset = combine_data(normal, active, low=0.5, high=0.0, flat=0.5)
    dataset = combine_all(normal, active)
    dataset = normalize_scale(dataset)
    dataset.to_pickle('./datasets/sets/dataset.pkl')

def deep_shallow_process(shallow_loc='./datasets/sets/dataset_shallow.pkl', deep_loc='./datasets/sets/dataset_deep.pkl'):
    active = sanitize(pd.read_pickle('./datasets/active/waves_full.pkl'), 30)
    #active = active.apply(lambda row: row[::2], axis=1)
    normal = sanitize(pd.read_pickle('./datasets/normal/waves_full.pkl'), 30)
    #normal = normal.apply(lambda row: row[::2], axis=1)

    #dataset_shallow = combine_all(normal, active, depth_class='shallow')
    #dataset_deep = combine_all(normal, active, depth_class='deep')
    #active = active[10000:]
    dataset_shallow, dataset_deep = combine_deep_shallow(normal, active, shuffle=True)

    dataset_shallow = normalize_scale(dataset_shallow)
    dataset_deep = normalize_scale(dataset_deep)

    #print(f"final sizes of {dataset_shallow.shape[0]}")

    dataset_deep.to_pickle(deep_loc)
    dataset_shallow.to_pickle(shallow_loc)

if __name__ == "__main__":
    deep_shallow_process()


