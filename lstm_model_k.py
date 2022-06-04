from datetime import datetime
from math import ceil

import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import tensorboard as tb
from tbparse import SummaryReader
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline, interp1d
from torchmetrics.functional import precision_recall

from lstm_dataset_k import TimeSeriesDataset, DownSample, LSTM, device, LossCounter

# Training parameters
n_epochs = 30 #100
batch_size = 32
test_size = 0.2
valid_size = 0.1
down_sample = 2
learning_rate = 0.001 #0.001

# Model parameters
input_size = ceil(3000 / down_sample)
hidden_size = 4 #2
num_classes = 1
num_layers = 1
shuffle = True
random_state = 42

def run(input_file='./datasets/sets/dataset.pkl', dataset=None, report_log=True, print_to_console=False):
    log_dir = "./runs/" + datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    writer = SummaryWriter(log_dir)
    # TODO shuffle (time-series), k-fold, sort based on time, trim recording
    # 0) Prepare data
    if not report_log:
        writer.close()
    if dataset is None:
        dataset = TimeSeriesDataset(input_file=input_file, transform=DownSample(down_sample))
    x_i, idx_test, y_i, _ = train_test_split(range(len(dataset)), dataset.y, stratify=dataset.y, random_state=random_state,
                                             test_size=test_size)
    idx_train, idx_valid, _, _ = train_test_split(x_i, y_i, stratify=y_i, random_state=random_state,
                                                  test_size=valid_size / (1 - test_size))
    train_split = Subset(dataset, idx_train)
    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=shuffle)
    valid_split = Subset(dataset, idx_valid)
    valid_loader = DataLoader(valid_split, batch_size=batch_size, shuffle=shuffle)
    test_split = Subset(dataset, idx_test)
    test_loader = DataLoader(test_split, batch_size=batch_size, shuffle=shuffle)

    # 1) Create model, loss and optimizer
    model = LSTM(input_size, hidden_size, num_classes, num_layers).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    writer.add_graph(model, iter(test_loader).next()[0])

    # 2) Training loop
    n_total_steps = len(train_loader)
    n_steps = n_total_steps
    train_counter = LossCounter(len(train_loader))
    valid_counter = LossCounter(len(valid_loader))
    for epoch in range(n_epochs):
        for i, (inp, labels) in enumerate(train_loader):
            labels = labels.unsqueeze(1)
            outputs = model(inp)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_counter.update(loss.item(), labels, outputs)
            if (i + 1) % n_steps == 0 and print_to_console:
                print(f'Epoch {epoch + 1}/{n_epochs}, step {i + 1}/{n_total_steps}, loss = {loss.item():.4f}')
        for i, (inp, labels) in enumerate(valid_loader):
            labels = labels.unsqueeze(1)
            outputs = model(inp)
            loss = criterion(outputs, labels)
            valid_counter.update(loss.item(), labels, outputs)
        writer.add_scalars('Loss',
                           {'train': train_counter.get_loss(), 'valid': valid_counter.get_loss()}, epoch + 1)
        writer.add_scalars('Accuracy',
                           {'train': train_counter.get_acc(), 'valid': valid_counter.get_acc()}, epoch + 1)

    # 3) Save results
    with torch.no_grad():
        all_labels = []
        all_predictions = []
        for i, (inp, labels) in enumerate(test_loader):
            labels = labels.unsqueeze(1)
            outputs = model(inp)
            all_labels.append(labels)
            all_predictions.append(torch.round(outputs))
        all_labels = torch.cat(all_labels).int()
        all_predictions = torch.cat(all_predictions).int()
        #accuracy = (all_predictions == all_labels).sum().item() / all_labels.shape[0]
        #t_p, t_r = precision_recall(all_predictions, all_labels, average='macro', num_classes=2)
        #precision = t_p.item()
        #recall = t_r.item()
        tp, fp, tn, fn = confusion(all_predictions, all_labels)

        accuracy = (tp+tn)/(tp+tn+fp+fn)
        precision = tp/(tp+fp)
        recall = tp / (tp + fn)

        print(tp,fp,tn,fn)
        print(f'Accuracy = {accuracy:.4f}')

        writer.add_pr_curve('pr_curve', all_labels, all_predictions)
        params = f"TRAINING PARAMETERS: " \
                 f"epochs: {n_epochs}, print_frequency: {n_steps}, batch: {batch_size}, lr: {learning_rate}, " \
                 f"train: {1 - test_size}, valid: {valid_size}, test: {test_size}, HZ: {100 / down_sample}, " \
                 f"seed: {random_state}, shuffle: {shuffle}. " \
                 f"MODEL PARAMETERS: " \
                 f"n_hidden: {hidden_size}, n_classes: {num_classes}, n_layers: {num_layers}. " \
                 f"RESULTS: accuracy: {accuracy}. "
        writer.add_text('Parameters', str(params))
        writer.close()

    if report_log:
        plot_history(log_dir)
    return [accuracy, precision, recall]

def confusion(prediction, truth):
    confusion_vector = prediction / truth

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives

def plot_history(log_dir):
    reader = SummaryReader(log_dir)
    df = reader.scalars
    print(df)
    plot_figure(df, "Accuracy")
    plot_figure(df, "Loss")

def plot_line(x, y):
    cubic_interploation_model = interp1d(x, y, kind="cubic")

    # Plotting the Graph
    #xnew = np.linspace(x.min(), x.max(), len(x)*5)
    #y_smooth = cubic_interploation_model(xnew)

    # create smooth line chart
    plt.plot(smooth(x, 0.6), y)

def smooth(scalars: list[float], weight: float) -> list[float]:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed
def plot_figure(df, metric):
    plt.figure()
    df = df[df['tag'] == metric]
    ax = plt.gca()
    train = df.iloc[::2]
    valid = df.iloc[1::2]
    #train['value'] = train['value'].ewm(span = 5).mean()
    plot_line(train['step'].to_numpy(), train['value'].to_numpy())
    plot_line(valid['step'].to_numpy(), valid['value'].to_numpy())
    #train.plot(x='step', y='value', kind='line', ax=ax)
    #valid.plot(x='step', y='value', kind='line', ax=ax)
    plt.title(f"Model {metric}")
    plt.ylabel(metric, fontsize="large")
    plt.xlabel("Total Number of Epochs", fontsize="large")
    plt.legend(["train", "val"], loc="best")
    plt.ylim(50, )
    plt.show()
    plt.close()

if __name__ == "__main__":
    run()