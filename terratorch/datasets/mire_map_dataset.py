# Copyright contributors to the Terratorch project

import glob
import os
from collections.abc import Sequence
from functools import lru_cache
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


# @lru_cache(maxsize=1024)
def load_netcdf(
    f: str | Path,
    labelvar: str,
    bands: tuple[str, ...],
    nan_replace: int | float | str | None = None,
    label_replace: int | None = -1,
    label_mapping: tuple[tuple[int, int], ...] | None = None,
    ignore_classes: tuple[int, ...] | None = None,
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

        # merge mire classes
        # lbl[lbl == 61] = 60
        # lbl[lbl == 62] = 60
        # lbl[lbl == 63] = 60
        # lbl[lbl == 64] = 60
        # lbl[lbl == 65] = 60



        if ignore_classes is not None:
            for k in ignore_classes:
                lbl[lbl == k] = label_replace

        if label_mapping is not None:
            for k, v in label_mapping:
                lbl[lbl == k] = v
        # print("unique labels", np.unique(lbl))

    return im, lbl


class MireMapDataset(NonGeoDataset):
    TEST_TILES = ["32VKL", "32VPR", "33WXT"] 
    VAL_TILES = ["32WPA", "33WVQ", "33WVS"]

    all_band_names = (
        "COASTAL_AEROSOL",
        "BLUE",
        "GREEN",
        "RED",
        "RED_EDGE_1",
        "RED_EDGE_2",
        "RED_EDGE_3",
        "NIR_BROAD",
        "NIR_NARROW",
        "WATER_VAPOR",
        # "CIRRUS",
        "SWIR_1",
        "SWIR_2",
    )

    band_mapping = {  # noqa: RUF012
        "B01": "COASTAL_AEROSOL",
        "B02": "BLUE",
        "B03": "GREEN",
        "B04": "RED",
        "B05": "RED_EDGE_1",
        "B06": "RED_EDGE_2",
        "B07": "RED_EDGE_3",
        "B08": "NIR_BROAD",
        "B09": "WATER_VAPOR",
        "B11": "SWIR_1",
        "B12": "SWIR_2",
        "B8A": "NIR_NARROW",
    }

    CLASS_NAMES = {  # noqa: RUF012
        10: "Bebygd areal",
        20: "Dyrka mark",
        26: "Dyrka mark med myr",
        30: "Skog",
        36: "Skog med myr",
        50: "Snaumark",
        56: "Snaumark med myr",
        60: "Myr",
        61: "Rismyr",           # "Myr med bebygd areal",
        62: "Bjønnskjeggmyr",   # "Myr med dyrka mark",
        63: "Grasmyr",          # "Myr med skog",
        64: "Blautmyr",         # "Myr med snaumark",
        65: "Starrsump",        # "Kombimyr",
        80: "Vann",
    }

    LABEL_MAP = {  # noqa: RUF012
        10: (0x66, 0x66, 0x66),  # Bebygd areal
        20: (0xFF, 0xEE, 0x00),  # Dyrka mark og beitevoller
        26: (0xBB, 0xAA, 0x00),  # Dyrka mark og beitevoller med innslag av myr
        30: (0x00, 0x88, 0x00),  # Skog
        36: (0x55, 0x88, 0x00),  # Skog med innslag av myr
        50: (0x00, 0xCC, 0x55),  # Snaumark
        56: (0x66, 0xAA, 0x00),  # Snaumark med innslag av myr
        60: (0x77, 0x77, 0x00),  # Myr
        61: (0x7A, 0x7A, 0x44),  # Myr med innslag av bebygd areal
        62: (0xAA, 0x99, 0x11),  # Myr med innslag av dyrka mark og beitevoller
        63: (0x77, 0x88, 0x00),  # Myr med innslag av skog
        64: (0x77, 0x77, 0x44),  # Kombinasjon av ulike myrtyper
        65: (0x88, 0xAA, 0x00),  # Myr med innslag av snaumark
        # 65: (0x88, 0xAA, 0x00),  # Myr med innslag av snaumark
        # 66: (0x77, 0x77, 0x44),  # Kombinasjon av ulike myrtyper
        80: (0x11, 0x00, 0xFF),  # Vann
    }
    IGNORE_CLASSES = [0, 26, 36, 56,        # noqa: RUF012
                      61, 62, 63, 64, 65, 66]   # merge mire classes to 60

    # MAJOR_RATIO = 0.7
    # CLASS_PROBS = {
    #     10: {10: 1.0},
    #     20: {20: 1.0},
    #     26: {20: MAJOR_RATIO, 60: 1.0 - MAJOR_RATIO},
    #     30: {30: 1.0},
    #     36: {30: MAJOR_RATIO, 60: 1.0 - MAJOR_RATIO},
    #     50: {50: 1.0},
    #     56: {50: MAJOR_RATIO, 60: 1.0 - MAJOR_RATIO},
    #     60: {60: 1.0},
    #     61: {60: MAJOR_RATIO, 10: 1.0 - MAJOR_RATIO},
    #     62: {60: MAJOR_RATIO, 20: 1.0 - MAJOR_RATIO},
    #     63: {60: MAJOR_RATIO, 30: 1.0 - MAJOR_RATIO},
    #     65: {60: MAJOR_RATIO, 50: 1.0 - MAJOR_RATIO},
    #     # 66: {66: 1.0}, # TODO: or ignore?
    #     80: {80: 1.0},
    # }

    NUM_CLASSES = len(LABEL_MAP) - len(IGNORE_CLASSES)

    rgb_bands = ("RED", "GREEN", "BLUE")

    BAND_SETS = {"all": all_band_names, "rgb": rgb_bands}  # noqa: RUF012

    def __init__(
        self,
        data_root: str,
        split="train",
        labelvar: str = "mire", #"mire_type",  # "mire",
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
        if split not in ["train", "test", "val", "predict"]:
            msg = "Split must be one of train, test, val."
            raise Exception(msg)

        self.data_root = Path(data_root)

        # self.rgb_indices = [self.all_band_names.index(b) for b in self.rgb_bands]
        self.rgb_indices = [0, 1, 2]

        img_files = glob.glob(str(self.data_root / "**/*.nc"), recursive=True)
        img_files = [f for f in img_files if not os.path.isdir(f)]
        if self.split == "val":
            # filter on validation tiles
            img_files = [f for f in img_files if self._filter_func(f, self.VAL_TILES)]
        elif self.split == "test":
            # filter on test tiles
            img_files = [f for f in img_files if self._filter_func(f, self.TEST_TILES)]
        elif self.split == "train":
            img_files = [f for f in img_files if not self._filter_func(f, self.VAL_TILES + self.TEST_TILES)]

        self.files = sorted(img_files)

        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
        self.use_metadata = use_metadata

        self.transform = transform if transform else default_transform

    def _filter_func(self, image_file, split):
        tile = self._get_tile(image_file)

        if tile in split:
            return True
        else:
            return False

    def _get_tile(self, file_path):
        # find the tile name from the file path
        path = Path(file_path)
        tile = path.parents[0].name.split("_")[-2].removeprefix("T")
        return tile

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        f = self.files[idx]
        im, lbl = load_netcdf(
            f,
            self.labelvar,
            tuple(self.band_netcdf_names),
            self.no_data_replace,
            self.no_label_replace,
            # {k: i for i, k in enumerate(sorted(set(self.LABEL_MAP.keys()) - set(self.IGNORE_CLASSES)))},
            tuple([(k, i) for i, k in enumerate(sorted(set(self.LABEL_MAP.keys()) - set(self.IGNORE_CLASSES)))]),
            tuple(self.IGNORE_CLASSES),
        )
        # print("img stats", im.shape, im.max(), im.min())
        # print(f, lbl.max(), lbl.min())
        # if lbl.max() == 255 and lbl.min() == 255:
        #     import pdb; pdb.set_trace()

        output = {
            "image": im,
            "mask": lbl,
            # "filename": f,
        }
        if self.transform:
            output = self.transform(**output)  # type: ignore
        return output

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
            num_classes=self.NUM_CLASSES,
            prediction=sample.get("prediction", None),
            suptitle=suptitle,
            class_names=[
                class_name for class_ind, class_name in self.CLASS_NAMES.items() if class_ind not in self.IGNORE_CLASSES
            ],
        )

    @staticmethod
    def _plot_sample(image, label, num_classes, prediction=None, suptitle=None, class_names=None):
        num_images = 5 if prediction is not None else 4
        fig, ax = plt.subplots(1, num_images, figsize=(8, 6), dpi=200)

        # for legend
        ax[0].axis("off")

        norm = colors.Normalize(vmin=0, vmax=num_classes)
        # cmap = colors.ListedColormap(
        #     [np.array(v) for k, v in MireMapDataset.LABEL_MAP.items() if k not in MireMapDataset.IGNORE_CLASSES]
        #     + [np.array([0, 0, 0])]
        # )
        ax[1].axis("off")
        ax[1].title.set_text("Image")
        ax[1].imshow(image)

        # TODO: fix
        label[label == 255] = num_classes

        print("unique labels", np.unique(label))

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
        legend_data = []
        for i, _ in enumerate(range(0, num_classes + 1)):
            if i < num_classes:
                class_name = class_names[i] if class_names else str(i)
            else:
                class_name = "Ignore"
            # class_name = class_names[i] if class_names and i < len(class_names) else str(i)
            data = [i, cmap(norm(i)), class_name]
            # data = [i, cmap(i), class_name]
            legend_data.append(data)
        handles = [Rectangle((0, 0), 1, 1, color=tuple(v for v in c)) for k, c, n in legend_data]
        labels = [n for k, c, n in legend_data]
        ax[0].legend(handles, labels, loc="center")
        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig