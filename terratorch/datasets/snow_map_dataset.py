# Copyright contributors to the Terratorch project

import glob
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import colors
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from torch import Tensor
from torchgeo.datasets import NonGeoDataset

from terratorch.datasets.utils import (
    clip_image_percentile,
    default_transform,
    validate_bands,
)


def load_netcdf(
    f: str | Path,
    labelvar: str,
    bands: list[str],
    nan_replace: int | float | str | None = None,
    label_replace: int | None = -1,
    label_mapping: dict[int, int] | None = None,
    ignore_classes: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    with xr.open_dataset(f, decode_coords="all") as ds:
        im = np.stack(
            [ds[k].transpose("y", "x").to_numpy().astype("float32") for k in bands],
            axis=-1,
        )
        mask = np.isnan(im)
        if nan_replace is not None:
            if isinstance(nan_replace, str):
                if nan_replace == "mean":
                    nan_replace = float(np.nanmean(im))
                elif nan_replace == "median":
                    nan_replace = float(np.nanmedian(im))
                else:
                    err_msg = f"Unknown nan_replace value: {nan_replace}"
                    raise ValueError(err_msg)

            im[mask] = nan_replace

        lbl = ds[labelvar].fillna(label_replace).astype("long")
        lbl = lbl.transpose("y", "x").to_numpy()
        lbl[mask.any(axis=-1)] = label_replace

        if ignore_classes is not None:
            for k in ignore_classes:
                lbl[lbl == k] = label_replace

        if label_mapping is not None:
            for k, v in label_mapping.items():
                lbl[lbl == k] = v
        # print("unique labels", np.unique(lbl))

    return im, lbl


class SnowMapDataset(NonGeoDataset):
    TEST_TILES = ['T33VMP', 'T33WVP', 'T33WWT', 'T33WMU', 'T33WXT', 
                  'T33VNM', 'T33WVR', 'T33VNP', 'T33VLL', 'T33WMT', 'T33WWP']
    VAL_TILES = ['T33WNU', 'T33WVN', 'T33VPP', 'T33WMS', 'T33VPQ', 'T33WVS',
                 'T33VNR', 'T33VMK', 'T33VDR', 'T33VLP', 'T33WEB', 'T33VUH']

    all_band_names = (
        "S1_reflectance_an",
        "S2_reflectance_an",
        "S3_reflectance_an",
        # "S4_reflectance_an",
        "S5_reflectance_an",
        "S6_reflectance_an",
        # "S7_BT_in",
        "S8_BT_in",
        "S9_BT_in",
    )
    band_mapping = {  # noqa: RUF012
        "B01": "S1_reflectance_an",
        "B02": "S2_reflectance_an",
        "B03": "S3_reflectance_an",
        # "B04": "S4_reflectance_an",
        "B04": "S5_reflectance_an",
        "B05": "S6_reflectance_an",
        # "B07": "S7_BT_in",
        "B08": "S8_BT_in",
        "B09": "S9_BT_in",
    }

    rgb_bands = ("RED", "GREEN", "BLUE")

    BAND_SETS = {"all": all_band_names, "rgb": rgb_bands}  # noqa: RUF012

    def __init__(
        self,
        data_root: str,
        split="train",
        labelvar: str = "scf",
        bands: Sequence[str] = BAND_SETS["all"],
        transform: A.Compose | None = None,
        constant_scale: float = 1.0,
        no_data_replace: float | str | None = 0,
        no_label_replace: int | None = -1,
        use_metadata: bool = False,  # noqa: FBT001, FBT002

    ):
        super().__init__()

        validate_bands(bands, self.all_band_names)
        self.bands = bands
        reversed_band_mapping = {v: k for k, v in self.band_mapping.items()}
        self.band_netcdf_names = [reversed_band_mapping[b] for b in bands]
        self.constant_scale = constant_scale
        self.data_root = Path(data_root)
        self.labelvar = labelvar
        self.split = split
        self.valid_mask = ["s3_valid_mask"]  # TODO read from config file
        self.target_bands = ["fsc"]
        if split not in ["train", "test", "val", "predict"]:
            msg = "Split must be one of train, test, val."
            raise Exception(msg)

        self.data_root = Path(data_root)

        # self.rgb_indices = [self.all_band_names.index(b) for b in self.rgb_bands]
        self.rgb_indices = [0, 1, 2]
        all_files = glob.glob(str(self.data_root / "**/*.npz"), recursive=True)
        if self.split == "val":
            # filter on validation tiles
            img_files = [f for f in all_files if self._filter_func(f, self.VAL_TILES)]
            # is_in_val = lambda f: any([year in str(f.zip.filename.split('/')[-1]) 
            #                                 for year in ['2019', '2020']])
            # img_files = self._filter_data(data_root, all_files, is_in_val)
        elif self.split == "test":
            # filter on test tiles
            img_files = [f for f in all_files if self._filter_func(f, self.TEST_TILES)]
            # is_in_test = lambda f: any([year in str(f.zip.filename.split('/')[-1]) 
            #                                 for year in ['2018']])
            # img_files = self._filter_data(data_root, all_files, is_in_test)
        elif self.split == "train":
            img_files = [f for f in all_files if not self._filter_func(f, self.VAL_TILES + self.TEST_TILES)]
            # is_in_train = lambda f: any([year in str(f.zip.filename.split('/')[-1]) 
            #                                 for year in ['2016', '2017']])
            # img_files = self._filter_data(data_root, all_files, is_in_train)

        self.files = sorted(img_files)

        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
        self.use_metadata = use_metadata

        self.transform = transform if transform else default_transform
        # from albumentations.pytorch import ToTensorV2
        # self.transform = A.Compose([A.CenterCrop(144, 144), ToTensorV2()])
        
    
    def _filter_data(self, data_root, files, is_in_training):
        files_to_use = []
        if filter is not None:
            for file in files:
                if file.endswith('.npz'):
                    path_to_file = os.path.join(data_root, file)
                    f = np.load(path_to_file)
                    if is_in_training(f):
                        file_ok = []
                        for band in list(self.bands) + self.valid_mask + self.target_bands:
                            if band in f.keys():
                                file_ok.append(True)
                            else:
                                file_ok.append(False)
                        if np.all(np.array(file_ok)):
                            files_to_use.append(file)
        return files_to_use

    def _filter_func(self, image_file, split):
        tile = self._get_tile(image_file)

        if tile in split:
            return True
        else:
            return False

    def _get_tile(self, file_path):
        # find the tile name from the file path
        path = Path(file_path)
        # tile = path.parents[0].name.split("_")[-2].removeprefix("T")
        tile = path.name.split("_")[0]
        return tile

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, item: int) -> dict[str, Any]:
        n = 0
        limit = 1000  # if not self.stop_on_errors else 1
        while n < limit:
            n += 1

            try:
                # if self.sample_randomly:
                # idx = np.random.randint(len(self.files))  # needed or is idx random already?
                idx = item % len(self.files)  # Wrap around the data if idx exceeds the actual data length
                
                return self._get_sample(idx)

            except Exception as e:
                # print(f'Could not load sample {self.files[idx]}')           
                self.files.pop(idx)
                if len(self.files) == 0:
                    raise Exception('No more samples left to draw from')
                # print(e)
                continue

        raise e

    def _get_sample(self, idx):
        
        sample = self.files[idx]
        
        # When sample is not loaded into memory in advance
        if type(sample) != dict:
            name = sample
            sample = {k: v for k, v in np.load(sample).items()}
            sample.update({"name": name})

        shape = sample["shape"][0]
        y, x = 0, 0
        ws = shape

        # Collect bands
        valid_mask = ["s3_valid_mask"]  # TODO read from config file
        target_bands = ["fsc"]  # TODO read from config file
        data = self._get_bands(sample, self.bands, y, x, ws).astype(np.float32)
        
        trgt = self._get_bands(sample, target_bands, y, x, ws)
        mask = self._get_bands(sample, valid_mask, y, x, ws)
        mask = np.sum(mask, -1, keepdims=True)

        # Maks and fill in missing values
        trgt[mask != 1] = 255  # np.nan
        trgt[np.sum(np.isnan(data), -1, keepdims=True) > 0] = 255  # np.nan
        trgt = trgt[:, :, 0]
        trgt[np.isnan(trgt)] = 255  

        # Replace nan values with non_replace strategy
        if self.no_data_replace is not None:
            if isinstance(self.no_data_replace, str):
                for b in range(0, data.shape[-1]):
                    im_b = data[..., b]
                    if self.no_data_replace == "mean":
                        no_data_replace = float(np.nanmean(im_b))
                    elif self.no_data_replace == "median":
                        no_data_replace = float(np.nanmedian(im_b))
                    else:
                        err_msg = f"Unknown no_data_replace value: {self.no_data_replace}"
                        raise ValueError(err_msg)
                    im_b[np.isnan(im_b)] = no_data_replace
        else:
            data[np.isnan(data)] = 0

        # Sample new data if data channels are empty or there are very few labelled sample.
        # if self.sample_randomly:
        if any(np.nansum(np.nansum(data, 0), 0) == 0) or np.sum(np.isfinite(mask))<100:
            return self[None]
        
        data = data.astype(np.float32)
        trgt = trgt.astype(np.float32)
     
        output = {
            "image": data,  # TODO: check if this work...
            "mask": trgt,
            # "filename": f,
        }
        if self.transform:
            output = self.transform(**output)  # type: ignore
        return output  #H x W x C

    def _get_bands(self, sample, bands, top, left, win_size, return_empty=False):
        """Reads bands from dict and crops"""

        data = []
        for b in bands:
            if return_empty and b not in sample:
                data.append(
                    np.zeros(win_size)
                )  # For multitarget setups when allow_empty_labels=True
            else:
                data.append(sample[b][top : top + win_size[0], left : left + win_size[1]])
        data = [np.expand_dims(d, -1) for d in data]
        data = np.concatenate(data, -1)

        return data

    def plot(self, sample: dict[str, Tensor], suptitle: str | None = None) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """

        image = sample["image"][self.rgb_indices, ...].permute(1, 2, 0).numpy()
        mask = sample["mask"].numpy()

        image = clip_image_percentile(image)

        return self._plot_sample(
            image,
            mask,
            prediction=sample.get("prediction", None),
            suptitle=suptitle,
        )

    @staticmethod
    def _plot_sample(image, label, prediction=None, suptitle=None):
        num_images = 5 if prediction is not None else 4
        fig, ax = plt.subplots(1, num_images, figsize=(8, 6))

        # for legend
        ax[0].axis("off")
        norm = colors.Normalize(vmin=0, vmax=255)
        # cmap = colors.ListedColormap(
        #     [np.array(v) for k, v in MireMapDataset.LABEL_MAP.items() if k not in MireMapDataset.IGNORE_CLASSES]
        #     + [np.array([0, 0, 0])]
        # )
        ax[1].axis("off")
        ax[1].title.set_text("Image")
        ax[1].imshow(image)

        # TODO: fix
        # label[label == 255] = num_classes

        # print("unique labels", np.unique(label))

        ax[2].axis("off")
        ax[2].title.set_text("Ground Truth Mask")
        ax[2].imshow(label, cmap="jet", norm=norm)
        # ax[2].imshow(label, cmap=cmap)  # , norm=norm)

        ax[3].axis("off")
        ax[3].title.set_text("GT Mask on Image")
        ax[3].imshow(image)
        ax[3].imshow(label, cmap="jet", alpha=0.3, norm=norm)
        # ax[3].imshow(label, cmap=cmap, alpha=0.3)  # , norm=norm)

        if prediction is not None:
            ax[4].title.set_text("Predicted Mask")
            ax[4].imshow(prediction, cmap="jet", norm=norm)
            # ax[4].imshow(prediction, cmap=cmap)  # , norm=norm)

        cmap = plt.get_cmap("jet")
        # legend_data = []
        # for i, _ in enumerate(range(0, num_classes + 1)):
        #     if i < num_classes:
        #         class_name = class_names[i] if class_names else str(i)
        #     else:
        #         class_name = "Ignore"
        #     # class_name = class_names[i] if class_names and i < len(class_names) else str(i)
        #     data = [i, cmap(norm(i)), class_name]
        #     # data = [i, cmap(i), class_name]
        #     legend_data.append(data)
        # handles = [Rectangle((0, 0), 1, 1, color=tuple(v for v in c)) for k, c, n in legend_data]
        # labels = [n for k, c, n in legend_data]
        # ax[0].legend(handles, labels, loc="center")
        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
