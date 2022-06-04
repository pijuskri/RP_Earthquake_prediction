#For finding the most accurate stations

import wave_download_k
import wave_process_k
import lstm_model_k

selected_stations = ['BFZ', 'BKZ', 'DCZ', 'DSZ', 'EAZ', 'HIZ', 'JCZ', 'KHZ', 'KNZ', 'KUZ', 'LBZ', 'LTZ', 'MLZ',
                         'MQZ', 'MRZ', 'MSZ', 'MWZ', 'MXZ', 'NNZ', 'ODZ', 'OPRZ', 'OUZ', 'PUZ', 'PXZ', 'QRZ', 'RPZ',
                         'SYZ', 'THZ', 'TOZ', 'TSZ', 'TUZ', 'URZ', 'VRZ', 'WCZ', 'WHZ', 'WIZ', 'WKZ', 'WVZ']

station = 'PUZ'
rerun_model = 5

wave_download_k.preprocess([station], True)
wave_process_k.process()
acc = 0
for i in range(rerun_model):
    acc += lstm_model_k.run(report_log=False)
acc = acc / rerun_model
print(f"{station}: {acc:.4f}")