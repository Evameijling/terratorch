import os
import torch
import torch.nn as nn
import warnings
from segmentation_models_pytorch.encoders import encoders as smp_encoders
import rasterio
import numpy as np
import terratorch
import time
from terratorch.datasets import HLSBands, SARBands
import lightning.pytorch as pl
from terratorch.models import PrithviModelFactory
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
import albumentations as A
from albumentations.pytorch import ToTensorV2  # Optional if you want to convert images to tensors directly
# from training_utils import CocoaMiningDataModule, CocoaMiningTask, FlopsCallback, GPUHoursCallback, PeakMemoryCallback, save_metrics
from terratorch.datamodules.cocoamining import CocoaMiningDataModule, CocoaMiningTask, FlopsCallback, GPUHoursCallback, PeakMemoryCallback, save_metrics

exp_id = '008'
BACKBONE="terramind_v1_base_tim"
TIM_MODALITIES = ["DEM"]
DECODER="UNetDecoder"
LOSS_FN='ce'
OPTIMIZER="AdamW"
LR=1e-3
BATCH_SIZE=4
EPOCHS=100
BANDS="S2"

results_folder = f'/home/egm/Data/Projects/FMs/CrossComparison/terratorch/examples/cocoamining/results'
csv_path = os.path.join(results_folder, 'metrics.csv')
os.makedirs(results_folder, exist_ok=True)

DATASET_PATH = '/home/egm/Data/Projects/FMs/CrossComparison/data/CocoaMiningDS/2016'
split_path = '/home/egm/Data/Projects/FMs/CrossComparison/data/CocoaMiningDS/2016/train_test_splits.csv'

ghana_mining_bands = [
    HLSBands.BLUE,
    HLSBands.GREEN,
    HLSBands.RED,
    HLSBands.RED_EDGE_1,
    HLSBands.RED_EDGE_2,
    HLSBands.RED_EDGE_3,
    HLSBands.NIR_BROAD,
    HLSBands.NIR_NARROW,
    HLSBands.SWIR_1,
    HLSBands.SWIR_2
]

#MEANS AND STDS FOR EACH BAND
means=[
        1465.16076660,
        1720.63488770,
        1695.57812500,
        2131.74975586,
        3223.77392578,
        3626.39038086,
        3727.01831055,
        3927.40429688,
        3205.23754883,
        2228.02368164,
        #-7.41195583,
        #-13.11112309,
        #192.80900574,
    ]
stds=[
        148.41740417,
        196.94464111,
        285.10070801,
        271.32940674,
        257.74014282,
        300.72171021,
        348.69989014,
        321.46240234,
        333.51858521,
        324.53616333,
        #1.37627256,
        #1.35524726,
        #13.54047298,
    ]


train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        ToTensorV2()
    ])


datamodule = CocoaMiningDataModule(
    batch_size=BATCH_SIZE,
    num_workers=8,
    train_data_root=DATASET_PATH,
    val_data_root=DATASET_PATH,
    test_data_root=DATASET_PATH,
    img_grep="IMAGE/IMG_*.tif",
    label_grep="MASK/MASK_*.tif",
    means=means,
    stds=stds,
    num_classes=3,
    
    train_transform=train_transform,
    #val_transform=val_transform,
    #test_transform=val_transform,

    dataset_bands=ghana_mining_bands,
    # Input bands of your model
    output_bands=ghana_mining_bands,
    no_data_replace=0,
    no_label_replace=-1,

    split_path=split_path,
)

model_args = {
        "backbone": BACKBONE, # see smp_encoders.keys()
        "backbone_pretrained": True,
        "backbone_modalities": ["S2L2A"],
        # "backbone_bands": {
        #     "S2L2A": ghana_mining_bands
        # },        
        "backbone_tim_modalities": TIM_MODALITIES,

        "necks": [
            {
                "name": "SelectIndices",
                "indices": [2, 5, 8, 11] # indices for terramind_v1_base
                # "indices": [5, 11, 17, 23] # indices for terramind_v1_large
            },
            {"name": "ReshapeTokensToImage",
             "remove_cls_token": False},  # TerraMind is trained without CLS token, which neads to be specified.
            {"name": "LearnedInterpolateToPyramidal"}  # Some decoders like UNet or UperNet expect hierarchical features. Therefore, we need to learn a upsampling for the intermediate embedding layers when using a ViT like TerraMind.
        ],

        "decoder": DECODER,
        "decoder_channels": [512, 256, 128, 64],

        # Head
        "head_dropout": 0.1,
        "num_classes": 3,
}


task = CocoaMiningTask(
    model_args=model_args,
    model_factory="EncoderDecoderFactory",
    loss=LOSS_FN,
    lr=LR,
    ignore_index=-1,
    optimizer=OPTIMIZER,
    optimizer_hparams={"weight_decay": 0.05},
    freeze_backbone=True,
    class_names=['Background', 'Mining', 'Cocoa']
    #class_weights=[0.1, 0.9]
)

keep_cols = torch.cat([
    torch.arange(1*256, 9*256),   # B2–B9 (BLUE … NIR_NARROW)
    torch.arange(10*256, 12*256)   # B11–B12 (SWIR1, SWIR2)
])

with torch.no_grad():
    emb = task.model.encoder.encoder_embeddings['untok_sen2l2a@224']  # ImageEncoderEmbedding
    old = emb.proj  # Linear(3072, 768, bias=False)
    assert isinstance(old, nn.Linear) and old.in_features == 3072 and old.out_features == 768

    new = nn.Linear(len(ghana_mining_bands)*256, 768, bias=False)  # (2560 -> 768)
    new.weight.copy_(old.weight[:, keep_cols])

    emb.proj = new
    for p in emb.proj.parameters():
        p.requires_grad = False

with torch.no_grad():
    emb = task.model.encoder.sampler.model.encoder_embeddings["untok_sen2l2a@224"]  # ImageEncoderEmbedding
    old = emb.proj  # Linear(3072, 768, bias=False)
    assert isinstance(old, nn.Linear) and old.in_features == 3072 and old.out_features == 768

    new = nn.Linear(len(ghana_mining_bands)*256, 768, bias=False)  # (2560 -> 768)
    new.weight.copy_(old.weight[:, keep_cols])

    emb.proj = new
    for p in emb.proj.parameters():
        p.requires_grad = False

datamodule.setup("fit")
checkpoint_callback = ModelCheckpoint(monitor=task.monitor, save_top_k=1, save_last=True)
early_stopping_callback = EarlyStopping(monitor=task.monitor, min_delta=0.00, patience=20)
logger = TensorBoardLogger(save_dir=results_folder, name=exp_id)
gpuHoursCb = GPUHoursCallback()
peakMemoryCb = PeakMemoryCallback()
flopsCb = FlopsCallback(input_shape=(1, len(ghana_mining_bands), 128, 128))

trainer = Trainer(
    devices=1, # Number of GPUs. Interactive mode recommended with 1 device
    precision="16-mixed",
    callbacks=[
        RichProgressBar(),
        checkpoint_callback,
        gpuHoursCb,
        peakMemoryCb,
        flopsCb,
        #early_stopping_callback,
        #LearningRateMonitor(logging_interval="epoch"),
    ],
    logger=logger,
    max_epochs=EPOCHS,
    default_root_dir=results_folder,
    log_every_n_steps=1,
    check_val_every_n_epoch=1
)

_ = trainer.fit(model=task, datamodule=datamodule)

train_metrics = trainer.callback_metrics
print("\n\nTraining metrics:")
for key, value in train_metrics.items():
    print(f"{key}: {value}")

res = trainer.test(model=task, datamodule=datamodule)

test_metrics = res[0]  # dict returned by trainer.test()
test_metrics["gpu_hours"] = gpuHoursCb.gpu_hours
test_metrics["peak_gb"] = peakMemoryCb.peak_gb
test_metrics["gflops"] = flopsCb.gflops

print("\n\nTest metrics:")
for key, value in test_metrics.items():
    print(f"{key}: {value}")

training_info = {
    "exp_id": exp_id,
    "bands": BANDS,
    "backbone": BACKBONE,
    "decoder": DECODER,
    "loss_fn": LOSS_FN,
    "optimizer": OPTIMIZER,
    "lr": LR,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "tim_modalities": TIM_MODALITIES,
}

save_metrics(training_info, test_metrics, train_metrics, csv_path)
