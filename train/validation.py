import torch 
from torchvision import transforms
from models.face_recogn.inceptionresnetV1 import InceptionResnetV1
from dataloader.Dataset import LFWDataset, TripletFaceDataset
from torch.nn.modules.distance import PairwiseDistance
import torch.optim as optim
import numpy as np
from torch.autograd import Function
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import auc
from sklearn.model_selection import KFold
from scipy import interpolate


def evaluate_lfw(distances, labels, num_folds=10, far_target=1e-3):
    
    # Calculate ROC metrics
    thresholds_roc = np.arange(0, 4, 0.01)
    true_positive_rate, false_positive_rate, precision, recall, accuracy, best_distances = \
        calculate_roc_values(
            thresholds=thresholds_roc, distances=distances, labels=labels, num_folds=num_folds
        )

    roc_auc = auc(false_positive_rate, true_positive_rate)

    # Calculate validation rate
    thresholds_val = np.arange(0, 4, 0.001)
    tar, far = calculate_val(
        thresholds_val=thresholds_val, distances=distances, labels=labels, far_target=far_target, num_folds=num_folds
    )

    return true_positive_rate, false_positive_rate, precision, recall, accuracy, roc_auc, best_distances,\
        tar, far


def calculate_roc_values(thresholds, distances, labels, num_folds=10):
    num_pairs = min(len(labels), len(distances))
    num_thresholds = len(thresholds)
    k_fold = KFold(n_splits=num_folds, shuffle=False)

    true_positive_rates = np.zeros((num_folds, num_thresholds))
    false_positive_rates = np.zeros((num_folds, num_thresholds))
    precision = np.zeros(num_folds)
    recall = np.zeros(num_folds)
    accuracy = np.zeros(num_folds)
    best_distances = np.zeros(num_folds)

    indices = np.arange(num_pairs)

    for fold_index, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the best distance threshold for the k-fold cross validation using the train set
        accuracies_trainset = np.zeros(num_thresholds)
        for threshold_index, threshold in enumerate(thresholds):
            _, _, _, _, accuracies_trainset[threshold_index] = calculate_metrics(
                threshold=threshold, dist=distances[train_set], actual_issame=labels[train_set]
            )
        best_threshold_index = np.argmax(accuracies_trainset)

        # Test on test set using the best distance threshold
        for threshold_index, threshold in enumerate(thresholds):
            true_positive_rates[fold_index, threshold_index], false_positive_rates[fold_index, threshold_index], _, _,\
                _ = calculate_metrics(
                    threshold=threshold, dist=distances[test_set], actual_issame=labels[test_set]
                )

        _, _, precision[fold_index], recall[fold_index], accuracy[fold_index] = calculate_metrics(
            threshold=thresholds[best_threshold_index], dist=distances[test_set], actual_issame=labels[test_set]
        )

        true_positive_rate = np.mean(true_positive_rates, 0)
        false_positive_rate = np.mean(false_positive_rates, 0)
        best_distances[fold_index] = thresholds[best_threshold_index]

    return true_positive_rate, false_positive_rate, precision, recall, accuracy, best_distances


def calculate_metrics(threshold, dist, actual_issame):
    # If distance is less than threshold, then prediction is set to True
    predict_issame = np.less(dist, threshold)

    true_positives = np.sum(np.logical_and(predict_issame, actual_issame))
    false_positives = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    true_negatives = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    false_negatives = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    # For dealing with Divide By Zero exception
    true_positive_rate = 0 if (true_positives + false_negatives == 0) else \
        float(true_positives) / float(true_positives + false_negatives)

    false_positive_rate = 0 if (false_positives + true_negatives == 0) else \
        float(false_positives) / float(false_positives + true_negatives)

    precision = 0 if (true_positives + false_positives) == 0 else\
        float(true_positives) / float(true_positives + false_positives)

    recall = 0 if (true_positives + false_negatives) == 0 else \
        float(true_positives) / float(true_positives + false_negatives)

    accuracy = float(true_positives + true_negatives) / dist.size

    return true_positive_rate, false_positive_rate, precision, recall, accuracy


def calculate_val(thresholds_val, distances, labels, far_target=1e-3, num_folds=10):
    num_pairs = min(len(labels), len(distances))
    num_thresholds = len(thresholds_val)
    k_fold = KFold(n_splits=num_folds, shuffle=False)

    tar = np.zeros(num_folds)
    far = np.zeros(num_folds)

    indices = np.arange(num_pairs)

    for fold_index, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the euclidean distance threshold that gives false acceptance rate (far) = far_target
        far_train = np.zeros(num_thresholds)
        for threshold_index, threshold in enumerate(thresholds_val):
            _, far_train[threshold_index] = calculate_val_far(
                threshold=threshold, dist=distances[train_set], actual_issame=labels[train_set]
            )
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds_val, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        tar[fold_index], far[fold_index] = calculate_val_far(
            threshold=threshold, dist=distances[test_set], actual_issame=labels[test_set]
        )

    return tar, far


def calculate_val_far(threshold, dist, actual_issame):
    # If distance is less than threshold, then prediction is set to True
    predict_issame = np.less(dist, threshold)

    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))

    num_same = np.sum(actual_issame)
    num_diff = np.sum(np.logical_not(actual_issame))

    if num_diff == 0:
        num_diff = 1
    if num_same == 0:
        return 0, 0

    tar = float(true_accept) / float(num_same)
    far = float(false_accept) / float(num_diff)

    return tar, far

import matplotlib.pyplot as plt
from sklearn.metrics import auc


def plot_roc_lfw(false_positive_rate, true_positive_rate, figure_name="roc.png"):
    """Plots the Receiver Operating Characteristic (ROC) curve.

    Args:
        false_positive_rate: False positive rate
        true_positive_rate: True positive rate
        figure_name (str): Name of the image file of the resulting ROC curve plot.
    """
    roc_auc = auc(false_positive_rate, true_positive_rate)
    fig = plt.figure()
    plt.plot(
        false_positive_rate, true_positive_rate, color='red', lw=2, label="ROC Curve (area = {:.4f})".format(roc_auc)
    )
    plt.plot([0, 1], [0, 1], color="blue", lw=2, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    fig.savefig(figure_name, dpi=fig.dpi)
    plt.close()


def plot_accuracy_lfw(log_file, epochs, figure_name="lfw_accuracies.png"):
    """Plots the accuracies on the Labeled Faces in the Wild dataset over the training epochs.

    Args:
        log_file (str): Path of the log file containing the lfw accuracy values to be plotted.
        epochs (int): Number of training epochs finished.
        figure_name (str): Name of the image file of the resulting lfw accuracies plot.
    """
    with open(log_file, 'r') as f:
        lines = f.readlines()
        epoch_list = [int(line.split('\t')[0]) for line in lines]
        accuracy_list = [round(float(line.split('\t')[1]), 2) for line in lines]

        fig = plt.figure()
        plt.plot(epoch_list, accuracy_list, color='red', label='LFW Accuracy')
        plt.ylim([0.0, 1.05])
        plt.xlim([0, epochs + 1])
        plt.xlabel('Epoch')
        plt.ylabel('LFW Accuracy')
        plt.title('LFW Accuracies plot')
        plt.legend(loc='lower right')
        fig.savefig(figure_name, dpi=fig.dpi)
        plt.close()



def validate_lfw(model, lfw_dataloader, model_architecture, epoch):
    model.eval()
    with torch.no_grad():
        l2_distance = PairwiseDistance(p=2)
        distances, labels = [], []

        print("Validating on LFW! ...")
        progress_bar = enumerate(tqdm(lfw_dataloader))

        for batch_index, (data_a, data_b, label) in progress_bar: # data_a : (batch, 3, 299, 299), label : (batch) true or false
            data_a = data_a.cuda()
            data_b = data_b.cuda()

            output_a, output_b = model(data_a), model(data_b)
            distance = l2_distance.forward(output_a, output_b)  # Euclidean distance (batch)

            distances.append(distance.cpu().detach().numpy())
            labels.append(label.cpu().detach().numpy())

        labels = np.array([sublabel for label in labels for sublabel in label]) # (num_batch* batch)
        distances = np.array([subdist for distance in distances for subdist in distance]) # (num_batch* batch)

        true_positive_rate, false_positive_rate, precision, recall, accuracy, roc_auc, best_distances, \
        tar, far = evaluate_lfw(
            distances=distances,
            labels=labels,
            far_target=1e-3
        )
        # Print statistics and add to log
        print("Accuracy on LFW: {:.4f}+-{:.4f}\tPrecision {:.4f}+-{:.4f}\tRecall {:.4f}+-{:.4f}\t"
              "ROC Area Under Curve: {:.4f}\tBest distance threshold: {:.2f}+-{:.2f}\t"
              "TAR: {:.4f}+-{:.4f} @ FAR: {:.4f}".format(
                    np.mean(accuracy),
                    np.std(accuracy),
                    np.mean(precision),
                    np.std(precision),
                    np.mean(recall),
                    np.std(recall),
                    roc_auc,
                    np.mean(best_distances),
                    np.std(best_distances),
                    np.mean(tar),
                    np.std(tar),
                    np.mean(far)
                )
        )
        
        log_dir = '/kaggle/working/logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        with open('/kaggle/working/logs/lfw_{}_log_triplet.txt'.format(model_architecture), 'a') as f:
            val_list = [
                epoch,
                np.mean(accuracy),
                np.std(accuracy),
                np.mean(precision),
                np.std(precision),
                np.mean(recall),
                np.std(recall),
                roc_auc,
                np.mean(best_distances),
                np.std(best_distances),
                np.mean(tar)
            ]
            log = '\t'.join(str(value) for value in val_list)
            f.writelines(log + '\n')

    try:

        if not os.path.exists('/kaggle/working/plots'):
            os.makedirs('/kaggle/working/plots/accuracies_plots')
            os.makedirs('/kaggle/working/plots/roc_plots')

        
        plot_roc_lfw(
            false_positive_rate=false_positive_rate,
            true_positive_rate=true_positive_rate,
            figure_name="/kaggle/working/plots/roc_plots/roc_{}_epoch_{}_triplet.png".format(model_architecture, epoch)
        )
        
        if not os.path.exists('/kaggle/working/logs'):
            os.makedirs('/kaggle/working/logs')
            
        plot_accuracy_lfw(
            log_file="/kaggle/working/logs/lfw_{}_log_triplet.txt".format(model_architecture),
            epochs=epoch,
            figure_name="/kaggle/working/plots/accuracies_plots/lfw_accuracies_{}_epoch_{}_triplet.png".format(model_architecture, epoch)
        )
    except Exception as e:
        print(e)

    return best_distances