import torch
import itertools
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report


def plot_confusion_matrix(cm, classes, filename,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.get_cmap('Blues')):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(15, 15))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.rcParams.update({'font.size': 28})

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filename + '.png', bbox_inches="tight")
    plt.close()


def predict_artist(y_pred, y_true, song_list,
                   class_names, ml_mode=False):
    """
    This function takes slices of songs and predicts their output.
    For each song, it votes on the most frequent artist.
    """
    # Obtain the list of songs
    songs = np.unique(song_list)

    prediction_list = []
    actual_list = []
    prediction_passed_num_list = []

    # Iterate through each song
    for song in songs:

        # Grab all slices related to a particular song
        predictions = y_pred[song_list == song]
        # predictions (#song, 20)

        actual = y_true[song_list == song]
        actual_list.append(stats.mode(actual, keepdims=False)[0])

        def trim_predictions(pred_, thresh_hold=0.5):
            if not ml_mode:
                # Get list of highest probability classes and their probability
                class_probability, class_prediction = torch.max(pred_, dim=-1)

                # keep only predictions confident about;
                pass_ids = class_probability > thresh_hold
                num_passed_ = sum(pass_ids)  # how many prediction over threshhold
                prediction_summary_trim = class_prediction[class_probability > thresh_hold]

                # deal with edge case where there is no confident class
                if len(prediction_summary_trim) == 0:
                    prediction_summary_trim = class_prediction
            else:
                prediction_summary_trim = pred_
                num_passed_ = pred_.shape[0]

            # get most frequent class
            new_prediction_ = stats.mode(prediction_summary_trim, keepdims=False)[0]

            # based on this threshold: final prediction with how many prediction
            return new_prediction_, num_passed_

        multi_threshold_predictions = []
        multi_threshold_passed_num = []
        for confidence_threshold in range(10):
            confidence_threshold = confidence_threshold / 10

            new_prediction, num_passed = trim_predictions(predictions, thresh_hold=confidence_threshold)

            # Keeping track of overall song classification accuracy
            multi_threshold_predictions.append(new_prediction)
            multi_threshold_passed_num.append(num_passed)

        prediction_list.append(multi_threshold_predictions)
        prediction_passed_num_list.append(multi_threshold_passed_num)

    # Print overall song accuracy
    actual_array = np.array(actual_list)  # (#song,)
    prediction_array = np.array(prediction_list)  # (#song, #thresholds(10))
    passed_slices_num_array = np.array(prediction_passed_num_list)  # (#song, #thresholds(10))
    prediction_array = np.transpose(prediction_array, [1, 0])  # (#thresholds(10), #song)
    passed_slices_num_array = np.transpose(passed_slices_num_array, [1, 0])  # (#thresholds(10), #song)

    prediction_results = []
    for i in range(10):
        cm = confusion_matrix(actual_array, prediction_array[i])

        class_report = classification_report(actual_array, prediction_array[i], digits=4,
                                             target_names=class_names, zero_division=1)

        class_report_dict = classification_report(actual_array, prediction_array[i], digits=4,
                                                  target_names=class_names, zero_division=1, output_dict=True)

        prediction_results.append((class_report, class_report_dict, cm))
    return prediction_results, passed_slices_num_array
