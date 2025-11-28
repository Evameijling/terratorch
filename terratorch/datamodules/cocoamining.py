import torch
from torch import nn
from torchmetrics import ClasswiseWrapper, MetricCollection
from terratorch.datamodules import GenericNonGeoSegmentationDataModule
from terratorch.tasks import SemanticSegmentationTask
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassJaccardIndex, MulticlassPrecision, MulticlassRecall
import re
import pandas as pd
import pytorch_lightning as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import measure_flops
import time
import os
import csv


class CocoaMiningDataModule(GenericNonGeoSegmentationDataModule):
    def __init__(self, split_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.split_path = split_path
        self.splits = pd.read_csv(self.split_path)
        self.splits["patch_name"] = self.splits["patch_name"].apply(lambda x: x.replace('MASK_', '').replace('_2016.tif', '').replace('_2022.tif', ''))
        train_files = self.splits[self.splits['split'] == 'train']['patch_name'].tolist()
        test_files = self.splits[self.splits['split'] == 'test']['patch_name'].tolist()

        self.train_img_grep = '|'.join(f'IMG_GH_{re.escape(patch)}' for patch in train_files)
        self.train_label_grep = '|'.join(f'MASK_{re.escape(patch)}' for patch in train_files)

        self.test_img_grep = '|'.join(f'IMG_GH_{re.escape(patch)}' for patch in test_files)
        self.test_label_grep = '|'.join(f'MASK_{re.escape(patch)}' for patch in test_files)


    def setup(self, stage: str) -> None:
        super().setup(stage)
        if stage in ["fit"]:
            self.train_dataset.image_files = [
                f for f in self.train_dataset.image_files if re.search(self.train_img_grep, f)
            ]
            self.train_dataset.segmentation_mask_files = [
                f for f in self.train_dataset.segmentation_mask_files if re.search(self.train_label_grep, f)
            ]
        if stage in ["fit", "validate"]:
            self.val_dataset.image_files = [
                f for f in self.val_dataset.image_files if re.search(self.test_img_grep, f)
            ]
            self.val_dataset.segmentation_mask_files = [
                f for f in self.val_dataset.segmentation_mask_files if re.search(self.test_label_grep, f)
            ]
        if stage in ["test"]:
            self.test_dataset.image_files = [
                f for f in self.test_dataset.image_files if re.search(self.test_img_grep, f)
            ]
            self.test_dataset.segmentation_mask_files = [
                f for f in self.test_dataset.segmentation_mask_files if re.search(self.test_label_grep, f)
            ]
        if stage in ["predict"] and self.predict_root:
            self.predict_dataset.image_files = [
                f for f in self.predict_dataset.image_files if re.search(self.test_img_grep, f)
            ]
            self.predict_dataset.segmentation_mask_files = [
                f for f in self.predict_dataset.segmentation_mask_files if re.search(self.test_label_grep, f)
            ]

class CocoaMiningTask(SemanticSegmentationTask):
    def configure_metrics(self) -> None:
        """Initialize the performance metrics."""
        num_classes: int = self.hparams["model_args"]["num_classes"]
        ignore_index: int = self.hparams["ignore_index"]
        class_names = self.hparams["class_names"]
        metrics = MetricCollection(
            {
                "Multiclass_Accuracy_Micro": MulticlassAccuracy(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    multidim_average="global",
                    average="micro",
                ),
                "Multiclass_Accuracy_Macro": MulticlassAccuracy(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    multidim_average="global",
                    average="macro",
                ),
                "Multiclass_Accuracy_Class": ClasswiseWrapper(
                    MulticlassAccuracy(
                        num_classes=num_classes,
                        ignore_index=ignore_index,
                        multidim_average="global",
                        average=None,
                    ),
                    labels=class_names,
                ),

                "Multiclass_Jaccard_Index_Micro": MulticlassJaccardIndex(
                    num_classes=num_classes, 
                    ignore_index=ignore_index, 
                    average="micro"
                ),
                "Multiclass_Jaccard_Index_Macro": MulticlassJaccardIndex(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="macro",
                ),
                "Multiclass_Jaccard_Index_Class": ClasswiseWrapper(
                    MulticlassJaccardIndex(num_classes=num_classes, ignore_index=ignore_index, average=None),
                    labels=class_names,
                ),
                
                "Multiclass_F1_Score_Micro": MulticlassF1Score(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    multidim_average="global",
                    average="micro",
                ),
                "Multiclass_F1_Score_Macro": MulticlassF1Score(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    multidim_average="global",
                    average="macro",
                ),
                "Multiclass_F1_Score_Class": ClasswiseWrapper(
                    MulticlassF1Score(num_classes=num_classes, ignore_index=ignore_index, average=None),
                    labels=class_names,
                ),

                "Multiclass_Precision_Micro": MulticlassPrecision(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    multidim_average="global",
                    average="micro",
                ),
                "Multiclass_Precision_Macro": MulticlassPrecision(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    multidim_average="global",
                    average="macro",
                ),
                "Multiclass_Precision_Class": ClasswiseWrapper(
                    MulticlassPrecision(num_classes=num_classes, ignore_index=ignore_index, average=None),
                    labels=class_names,
                ),

                "Multiclass_Recall_Micro": MulticlassRecall(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    multidim_average="global",
                    average="micro",
                ),
                "Multiclass_Recall_Macro": MulticlassRecall(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    multidim_average="global",
                    average="macro",
                ),
                "Multiclass_Recall_Class": ClasswiseWrapper(
                    MulticlassRecall(num_classes=num_classes, ignore_index=ignore_index, average=None),
                    labels=class_names,
                ),


            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        if self.hparams["test_dataloaders_names"] is not None:
            self.test_metrics = nn.ModuleList(
                [metrics.clone(prefix=f"test/{dl_name}/") for dl_name in self.hparams["test_dataloaders_names"]]
            )
        else:
            self.test_metrics = nn.ModuleList([metrics.clone(prefix="test/")])

            
class GPUHoursCallback(Callback):
    def __init__(self):
        super().__init__()
        self.gpu_hours = None
    def on_fit_start(self, trainer, pl_module):
        torch.cuda.synchronize()
        self._t0 = time.time()

    def on_fit_end(self, trainer, pl_module):
        torch.cuda.synchronize()
        t1 = time.time()
        duration = t1 - self._t0                     # seconds of real time
        num_gpus = max(1, trainer.strategy.world_size)
        gpu_hours = duration * num_gpus / 3600.0      # GPU-seconds → GPU-hours
        self.gpu_hours = gpu_hours
        print(f"\n→ GPU-Hours: {gpu_hours:.3f} h")

class PeakMemoryCallback(Callback):
    def __init__(self):
        super().__init__()
        self.peak_gb = None
    def on_fit_start(self, trainer, pl_module):
        torch.cuda.reset_peak_memory_stats()

    def on_fit_end(self, trainer, pl_module):
        peak_bytes = torch.cuda.max_memory_allocated()      # bytes
        peak_gb = peak_bytes / (1024**3)
        self.peak_gb = peak_gb
        print(f"→ Peak GPU memory: {peak_gb:.2f} GB")
        

class FlopsCallback(Callback):
    """
    Computes and logs model FLOPs at the end of training.
    Usage: add FlopsEndCallback(input_shape=(1,6,128,128)) to your Trainer callbacks.
    """
    def __init__(self, input_shape=(1, 6, 128, 128)):
        super().__init__()
        self.input_shape = input_shape
        self.gflops = None

    def on_fit_end(self, trainer, pl_module):   
        x = torch.randn(self.input_shape, device=pl_module.device)  # Adjust input shape as needed

        pl_module.model.eval()  # Set the model to evaluation mode
        model_fwd = lambda: pl_module.model(x)
        fwd_flops = measure_flops(pl_module.model, model_fwd)
        gflops = fwd_flops / 1e9
        print(f"→ FLOPs: {gflops:.3f} GFLOPs")
        self.gflops = gflops

def save_metrics(training_info, test_metrics, train_metrics, csv_path):
    tim_modalities_txt = ""
    if len(training_info["tim_modalities"]) == 0 or training_info["tim_modalities"] == "-":
        tim_modalities_txt = "-"
    else:
        tim_modalities_txt = "+".join(training_info["tim_modalities"])
    row = {
        'exp_id': training_info["exp_id"],
        'bands': training_info["bands"],
        'backbone': training_info["backbone"],
        'tim_modalities': tim_modalities_txt,
        'decoder': training_info["decoder"],
        'loss_fn': training_info["loss_fn"],
        'optimizer': training_info["optimizer"],
        'lr': training_info["lr"],
        'batch_size': training_info["batch_size"],
        'epochs': training_info["epochs"],
        # training metrics
        'train_loss': train_metrics['train/loss'].item(),
        'train_micro_iou': train_metrics['train/Multiclass_Jaccard_Index_Micro'].item(),
        'train_macro_iou': train_metrics['train/Multiclass_Jaccard_Index_Macro'].item(),
        'train_micro_f1': train_metrics['train/Multiclass_F1_Score_Micro'].item(),
        'train_macro_f1': train_metrics['train/Multiclass_F1_Score_Macro'].item(),
        # test metrics
        'test_loss': test_metrics['test/loss'],
        # averages
        'micro_precision': test_metrics['test/Multiclass_Precision_Micro'],
        'macro_precision': test_metrics['test/Multiclass_Precision_Macro'],
        'micro_recall':    test_metrics['test/Multiclass_Recall_Micro'],
        'macro_recall':    test_metrics['test/Multiclass_Recall_Macro'],
        'micro_f1':        test_metrics['test/Multiclass_F1_Score_Micro'],
        'macro_f1':        test_metrics['test/Multiclass_F1_Score_Macro'],
        'micro_iou':       test_metrics['test/Multiclass_Jaccard_Index_Micro'],
        'macro_iou':       test_metrics['test/Multiclass_Jaccard_Index_Macro'],
        # class‐wise metrics
        'precision_Background': test_metrics['test/multiclassprecision_Background'],
        'recall_Background':    test_metrics['test/multiclassrecall_Background'],
        'iou_Background':       test_metrics['test/multiclassjaccardindex_Background'],
        'f1_Background':        test_metrics['test/multiclassf1score_Background'],
        'precision_Mining':     test_metrics['test/multiclassprecision_Mining'],
        'recall_Mining':        test_metrics['test/multiclassrecall_Mining'],
        'iou_Mining':           test_metrics['test/multiclassjaccardindex_Mining'],
        'f1_Mining':            test_metrics['test/multiclassf1score_Mining'],
        'precision_Cocoa':      test_metrics['test/multiclassprecision_Cocoa'],
        'recall_Cocoa':         test_metrics['test/multiclassrecall_Cocoa'],
        'iou_Cocoa':            test_metrics['test/multiclassjaccardindex_Cocoa'],
        'f1_Cocoa':             test_metrics['test/multiclassf1score_Cocoa'],
        # additional metrics
        'gpu_hours': test_metrics["gpu_hours"],
        'peak_memory': test_metrics["peak_gb"],
        'gflops': test_metrics["gflops"]
    }

    # append (or create) the CSV
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Appended metrics to {csv_path}")