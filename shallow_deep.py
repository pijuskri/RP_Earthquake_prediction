#For finding the most accurate stations
import pandas as pd

import wave_download_k
import wave_process_k
import lstm_model_k
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from lstm_dataset_k import TimeSeriesDataset, DownSample


def define_box_properties(plot_name, color_code, label):
    for k, v in plot_name.items():
        plt.setp(plot_name.get(k), color=color_code)

    # use plot function to draw a small line to name the legend.
    plt.plot([], c=color_code, label=label)
    plt.legend()

def plot_figure(shallow, deep):
    shallow_plot = plt.boxplot(shallow, positions=np.array(np.arange(len(shallow))) * 2.0 - 0.35,widths=0.6)
    deep_plot = plt.boxplot(deep, positions=np.array(np.arange(len(deep))) * 2.0 + 0.35,widths=0.6)

    define_box_properties(shallow_plot, '#D7191C', 'Shallow')
    define_box_properties(deep_plot, '#2C7BB6', 'Deep')

    #plt.figure()
    #ax = plt.gca()
    #ax.boxplot([shallow, deep])
    plt.title(f"Boxplot of deep and shallow model accuracy")
    plt.ylabel('Accuracy', fontsize="large")
    plt.xticks([1, 2], ["Shallow", "Deep"])
    plt.xlim(-2, 4)
    # set the limit for y axis
    #plt.ylim(0, 100)
    plt.show()
    plt.close()

def plot_sns(shallow, deep):
    shallow = np.array(shallow)
    deep = np.array(deep)
    df = pd.DataFrame(np.concatenate((shallow, deep)), columns=["Accuracy", "Precision", "Recall"])
    depth_labels = []
    for i in range(len(shallow)):
        depth_labels.append("shallow")
    for i in range(len(deep)):
        depth_labels.append("deep")
    df['Depth'] = depth_labels
    #df['Value'] = np.zeros((len(shallow)+len(deep)))
    df = pd.melt(df, id_vars=["Depth"], value_vars=["Accuracy", "Precision", "Recall"])
    sns.boxplot(x=df['Depth'], y=df['value'], hue=df['variable'])
    plt.title(f"Deep and shallow data performance metrics")
    plt.ylabel('Metric value', fontsize="large")
    plt.grid(axis='y')
    plt.legend(bbox_to_anchor=(0.7, 0.22))
    plt.show()
    plt.close()

#selected_stations = ['KHZ', 'MXZ', 'VRZ', 'WIZ', 'BFZ', 'PUZ', 'NNZ']
selected_stations = ['BFZ', 'BKZ', 'DCZ', 'DSZ', 'HIZ', 'JCZ', 'KHZ', 'KUZ', 'LBZ',
                        'MSZ', 'MWZ', 'MXZ', 'NNZ', 'ODZ', 'OPRZ', 'OUZ', 'PUZ', 'PXZ', 'QRZ', 'RPZ',
                         'SYZ', 'THZ', 'TOZ', 'URZ', 'VRZ', 'WIZ', 'WKZ', 'WVZ']

#selected_stations = ['MRZ']

shallow_loc='./datasets/sets/dataset_shallow.pkl'
deep_loc='./datasets/sets/dataset_deep.pkl'

#wave_download_k.preprocess(selected_stations, True)
#wave_process_k.deep_shallow_process(shallow_loc, deep_loc)
print("done with processing")
#lstm_model_k.run(input_file=shallow_loc)
#lstm_model_k.run(input_file=deep_loc)


def get_avg_acc():
    out_shallow = []
    out_deep = []
    dataset_shallow = TimeSeriesDataset(input_file=shallow_loc, transform=DownSample(2))
    dataset_deep = TimeSeriesDataset(input_file=deep_loc, transform=DownSample(2))
    print("finished dataset loading")
    print("dataset sizes", len(dataset_shallow), len(dataset_deep))
    for i in range(10):
        print(f"Doing run {i}")
        out_shallow.append(lstm_model_k.run(input_file=shallow_loc, dataset=dataset_shallow, report_log=False))
        out_deep.append(lstm_model_k.run(input_file=deep_loc, dataset=dataset_deep, report_log=False))

    print(out_shallow, out_deep)

    plot_figure(out_shallow, out_deep)

    out_shallow = np.array(out_shallow).transpose()
    out_deep = np.array(out_deep).transpose()

    print(out_shallow, out_deep)

    #out_shallow.tofile("shallow_run.txt", ',')
    #out_deep.tofile("deep_run.txt", ',')
    np.save('shallow_runs.npy', out_shallow)
    np.save('deep_runs.npy', out_deep)

    #plot_figure(out_shallow, out_deep)

def mean_info(data):
    data = np.array(data)
    print(np.std(data, axis=0))
    print(np.mean(data, axis=0))
    #print(np.std(out_deep[0]))
    #print(np.mean(out_shallow[0]))
    #print(np.mean(out_deep[0]))

#get_avg_acc()

#acc_shallow = [0.9092039800995025, 0.8805970149253731, 0.9036069651741293, 0.8818407960199005, 0.9098258706467661, 0.9197761194029851, 0.9166666666666666, 0.9067164179104478, 0.9110696517412935, 0.904228855721393]
#acc_deep = [0.8917910447761194, 0.8911691542288557, 0.8774875621890548, 0.9228855721393034, 0.875, 0.931592039800995, 0.9017412935323383, 0.8986318407960199, 0.8905472636815921, 0.8588308457711443]

#[0.9092039800995025, 0.8805970149253731, 0.9036069651741293, 0.8818407960199005, 0.9098258706467661, 0.9197761194029851, 0.9166666666666666, 0.9067164179104478, 0.9110696517412935, 0.904228855721393] [0.8917910447761194, 0.8911691542288557, 0.8774875621890548, 0.9228855721393034, 0.875, 0.931592039800995, 0.9017412935323383, 0.8986318407960199, 0.8905472636815921, 0.8588308457711443]


shallow_metrics = [[0.9421461897356143, 0.9293051359516616, 0.9570628500311139], [0.8345256609642302, 0.7981142540210759, 0.895457373988799], [0.8951788491446345, 0.9355281207133059, 0.8487865588052271], [0.8566096423017108, 0.8720779220779221, 0.8357187305538271], [0.9097978227060654, 0.8752136752136752, 0.955818294959552], [0.929393468118196, 0.9383735705209657, 0.9191039203484754], [0.9337480559875583, 0.9462227912932138, 0.9197261978842564], [0.9200622083981338, 0.901307966706302, 0.9433727442439328], [0.8945567651632971, 0.8773809523809524, 0.9172370877411326], [0.9278382581648522, 0.9220380601596071, 0.9346608587429994]]
deep_metrics = [[0.8653188180404354, 0.8452941176470589, 0.8942128189172371], [0.9175738724727839, 0.9116564417177914, 0.9247044181705041], [0.8814930015552099, 0.897020725388601, 0.8618543870566272], [0.9104199066874028, 0.9441077441077441, 0.8724331051649036], [0.9200622083981338, 0.895662368112544, 0.9508400746733043], [0.9241057542768274, 0.9272727272727272, 0.9203484754200373], [0.913841368584759, 0.8958333333333334, 0.9365276913503423], [0.8818040435458787, 0.9305263157894736, 0.8251400124455507], [0.8808709175738725, 0.8700120918984281, 0.895457373988799], [0.8755832037325039, 0.9363702096890817, 0.805849408836341]]
mean_info(shallow_metrics)
mean_info(deep_metrics)

plot_sns(shallow_metrics, deep_metrics)