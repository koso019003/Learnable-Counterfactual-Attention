import os
import torch
import warnings
import lightning
import numpy as np

from pytorch_lightning.utilities.model_summary import ModelSummary
from sklearn.metrics import confusion_matrix, classification_report

from src.dataset import GTZANDataModule
from src.model import TimeSeqMoSEMERTModel
from src.utility import plot_confusion_matrix, predict_artist


def test_model(
    model_config,
    train_config,
    data_config,
    save_folder,
    model_weight,
):
    """
    Main function for training the model and testing
    """
    # Load data
    print("Loading dataset...")
    music_data = GTZANDataModule(
        train_config["pre_trained_model"],
        data_config=data_config,
        batch_size=train_config["batch_size"],
    )
    music_data.setup(stage="test")
    
    # Create model
    if not train_config["get_cf_result"]:
        model_config["mose_cf"] = 0
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = TimeSeqMoSEMERTModel.load_from_checkpoint(
            model_weight, map_location=torch.device("cpu"), config=model_config, strict=False
        )
    model.eval()
    model.set_forward("test")

    # Generate result
    ## Predict the test dataset
    print("Predicting test dataset result")
    trainer = lightning.Trainer(default_root_dir=save_folder, logger=False)
    pred = trainer.predict(
        model,
        dataloaders=music_data.test_dataloader(),
        return_predictions=True
    )
    y_pred, y_cfs = aggregate_pred(pred)
    
    ## Save and show results
    base_filename = os.path.join(save_folder, "testing_result")
    os.makedirs(base_filename, exist_ok=True)

    print(f"\nNumber of parameters: {ModelSummary(model).total_parameters / 10**6 :.2f} (M)")
    write_test_result(y_pred, music_data, os.path.join(base_filename, "y"))
    if y_cfs is not None:
        write_test_result(
            y_cfs, music_data, os.path.join(base_filename, "y_cf")
        )

def aggregate_pred(list_pred):
    y_pred_ = []
    y_cf_ = []
    for batch_pred in list_pred:
        y_pred_.append(batch_pred[0])
        if batch_pred[1] is not None:
            y_cf_.append(batch_pred[1])
    if y_cf_:
        return torch.cat(y_pred_, dim=0), torch.cat(y_cf_, dim=0)
    return torch.cat(y_pred_, dim=0), None

def write_test_result(y_score, data_module, filename):
    y_true = data_module.test_set.y
    song_list = data_module.test_set.s
    class_names_original = data_module.ids2class(np.arange(data_module.nb_classes))

    y_predict = torch.argmax(y_score, dim=-1)

    # Calculate and plot confusion matrix
    cm = confusion_matrix(y_true, y_predict)
    plot_confusion_matrix(
        cm,
        classes=class_names_original,
        filename=filename + "_frames_confusion_matrix",
        normalize=True,
        title="",
    )

    scores = classification_report(
        y_true, y_predict, target_names=class_names_original, zero_division=1, digits=4
    )
    scores_dict = classification_report(
        y_true,
        y_predict,
        target_names=class_names_original,
        zero_division=1, 
        digits=4,
        output_dict=True,
    )

    # Predict artist using pooling methodology
    # (#branchs, #threshold, 1)
    multi_threshold_predictions, multi_threshold_passed_slices_num = predict_artist(
        y_pred=y_score,
        y_true=y_true,
        song_list=song_list,
        class_names=class_names_original,
    )
    pooling_max_acc, pooling_max_acc_idx = 0, 0
    for j in range(len(multi_threshold_predictions)):
        if multi_threshold_predictions[j][1]["accuracy"] > pooling_max_acc:
            pooling_max_acc = multi_threshold_predictions[j][1]["accuracy"]
            pooling_max_acc_idx = j
    
    # Print out and save metrics
    num_test = len(data_module.test_set.y)
    num_songs = len(np.unique(data_module.test_set.s))
    
    print("Test accuracy:")
    print(f'Frame level: {scores_dict["accuracy"] :.2f}\tSong level: {pooling_max_acc :.2f}')

    with open(filename + ".txt", "w") as f:
        f.write(f"\nTesting on {num_test} slices")
        f.write(
            "\nnb_classes: "
            + str(data_module.nb_classes)
            + "\nslice_length: "
            + str(data_module.slice_length)
        )
        f.write("\nTest accuracy: " + str(scores_dict["accuracy"]))
        f.write("\nTest results on each slice:\n")
        f.write(str(scores))
        f.write(f"\n\nScores when pooling song slices in {num_songs} songs:\n")
        
        f.write(f"Threshold: 0.{pooling_max_acc_idx}\n")
        f.write(f"Passing frames: {sum(multi_threshold_passed_slices_num[pooling_max_acc_idx])}\n")
        f.write(f"Polling accuracy: {pooling_max_acc}\n")

        f.write("\n")
        f.write(str(multi_threshold_predictions[pooling_max_acc_idx][0]))
