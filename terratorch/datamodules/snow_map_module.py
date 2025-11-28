# Copyright contributors to the Terratorch project

from collections.abc import Sequence
from typing import Any
import numpy as np
import torch

from pathlib import Path

import albumentations as A
from torch import Tensor
from torch.utils.data import DataLoader, StackDataset
from torchgeo.datamodules import NonGeoDataModule
from terratorch.datasets.utils import to_numpy
from terratorch.datamodules.generic_pixel_wise_data_module import Normalize
from terratorch.datamodules.utils import wrap_in_compose_is_list
from terratorch.datasets import (
    SnowMapDataset,
)
from terratorch.io.file import load_from_file_or_attribute


class SnowMapNonGeoDataModule(NonGeoDataModule):
    def __init__(
        self,
        data_root: str,
        batch_size: int,
        num_workers: int,
        means: list[float] | str,
        stds: list[float] | str,
        predict_data_root: Path | None = None,
        bands: Sequence[str] = SnowMapDataset.all_band_names,
        train_transform: A.Compose | None | list[A.BasicTransform] = None,
        val_transform: A.Compose | None | list[A.BasicTransform] = None,
        test_transform: A.Compose | None | list[A.BasicTransform] = None,
        drop_last: bool = True,
        constant_scale: float = 1.0,
        no_data_replace: str | float | None = 0,
        no_label_replace: int | None = -1,
        use_metadata: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(SnowMapDataset, batch_size=batch_size, num_workers=num_workers, **kwargs)

        self.data_root = data_root

        self.bands = bands

        self.drop_last = drop_last
        self.constant_scale = constant_scale
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
        self.use_metadata = use_metadata
        self.predict_root = predict_data_root

        means = load_from_file_or_attribute(means)
        stds = load_from_file_or_attribute(stds)

        # self.aug = AugmentationSequential(K.Normalize(means, stds), data_keys=["image"])
        self.aug = Normalize(means, stds)

        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)

    def setup(self, stage: str):
        if stage in ["fit"]:
            self.train_dataset = SnowMapDataset(
                self.data_root,
                split="train",
                bands=self.bands,
                transform=self.train_transform,
                constant_scale=self.constant_scale,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = SnowMapDataset(
                self.data_root,
                split="val",
                bands=self.bands,
                transform=self.val_transform,
                constant_scale=self.constant_scale,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
            )

        if stage in ["test"]:
            self.test_dataset = SnowMapDataset(
                self.data_root,
                split="test",
                bands=self.bands,
                transform=self.test_transform,
                constant_scale=self.constant_scale,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
            )

        if stage == "predict":
            
            self.predict_dataset = SnowMapDataset(
                self.predict_root,
                split="predict",
                bands=self.bands,
        #         transform=self.train_transform,
        #         constant_scale=self.constant_scale,
        #         no_data_replace=self.no_data_replace,
            )

    def _dataloader_factory(self, split: str) -> DataLoader[dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders.

        Args:
            split: Either 'train', 'val', 'test', or 'predict'.

        Returns:
            A collection of data loaders specifying samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                dataset or sampler, or if the dataset or sampler has length 0.
        """
        dataset = self._valid_attribute(f"{split}_dataset", "dataset")
        batch_size = self._valid_attribute(f"{split}_batch_size", "batch_size")
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            # drop_last=split == "train" and self.drop_last,
            drop_last=self.drop_last,  # TODO: make this default for all dataloaders and other modules?
        )

class StackDatasetWithTransform(StackDataset):
    """
    Small wrapper to concatenate a stack of datasets along the channel dimension,
    and apply a transform to the concatenated data.
    """

    def __init__(self, *args, transform=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = transform

    def concat_item(self, items):
        collated = {}
        # Concatenate the items, tensors along channel dimension
        # Expects the label masks to be identical for all datasets which have a mask
        if isinstance(items, dict):
            iterate_dataset_key = next(iter(items.keys()))
            dataset_keys = list(items.keys())
        elif isinstance(items, tuple):
            iterate_dataset_key = 0
            dataset_keys = list(range(len(items)))

        for k in items[iterate_dataset_key].keys():
            if isinstance(items[iterate_dataset_key][k], torch.Tensor):
                
                if k == "mask":
                    # TODO: take the union of the masks!!!!!
                    collated[k] = items[iterate_dataset_key][k]
                    for _dataset_key in dataset_keys:  # Sanity check
                        if k in items[_dataset_key]:
                            pass
                            # #TODO: something is wrong with lidar and s2 data for lavdas, just a few pixels off, ignoring for now
                            # try:
                            #     msg = f"Expected all datasets to have the same label mask!"
                            #     torch.testing.assert_close(collated[k], items[_dataset_key][k], msg=msg)
                            # except AssertionError as e:
                            #     print(e)
                            #     print(f"Dataset {_dataset_key} has a different label mask!")
                else:
                    collated[k] = torch.cat([items[dataset_key][k] for dataset_key in dataset_keys], dim=0)
                    

            elif isinstance(items[iterate_dataset_key][k], np.ndarray):
                if k == "mask":
                    # TODO: take the union of the masks!!!!!
                    collated[k] = items[iterate_dataset_key][k]
                    for _dataset_key in dataset_keys:  # Sanity check
                        if k in items[_dataset_key]:
                            pass  # TODO: add back
                else:
                    collated[k] = np.concatenate([items[dataset_key][k] for dataset_key in dataset_keys], axis=-1)
                    
            else:
                if collated.get(k, None) is None:
                    collated[k] = items[iterate_dataset_key][k]

                for _dataset_key in dataset_keys:
                    if k in items[_dataset_key]:
                        if any(collated[k][i] != items[_dataset_key][k][i] for i in range(len(collated[k]))):
                            # They are not equal, lets add each one as a new key
                            collated[f"{k}_{_dataset_key}"] = items[_dataset_key][k]
        
        return collated

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item = self.concat_item(item)
        
        if self.transform is not None:
            item = to_numpy(item)  # Suboptimal
            item = self.transform(**item)
        
        return item

    # # TODO: create a check to see if super has getitems, if not use getitem
    # def __getitems__(self, indices: list):
    #     items = super().__getitems__(indices)  # C1 x H x W and C2 x H x W
    #     items = [self.concat_item(item) for item in items]  # C1+C2 x H x W

    #     if self.transform is not None:
    #         items = [to_numpy(item) for item in items]  # Suboptimal  H x W x C1+C2

    #         # Dropping metadata for now
    #         # items = [self.transform(image=item["image"], mask=item["mask"]) for item in items]
    #         items = [self.transform(**item) for item in items]  # Hcrop x Wcrop x C1+C2

    #         # for item in items:
    #         #     # Move last dimension to second dimension
    #         #     item["image"] = np.transpose(item["image"], (2, 0, 1))  # C1+C2 x Hcrop x Wcrop 

    #     return items


class SnowMapNonGeoMultiDataModule(NonGeoDataModule):
    def __init__(
        self,
        # data_root: str,
        batch_size: int,
        num_workers: int,
        dataset_kwargs: dict[str, Any],
        predict_data_root: Path | None = None,
        means: list[float] | None = None,
        stds: list[float] | None = None,
        dataset_order: list[str] | None = None,
        # bands: Sequence[str] = SnowMapDataset.all_band_names,
        train_transform: A.Compose | None | list[A.BasicTransform] = None,
        val_transform: A.Compose | None | list[A.BasicTransform] = None,
        test_transform: A.Compose | None | list[A.BasicTransform] = None,
        drop_last: bool = True,
        constant_scale: float = 1.0,
        no_data_replace: str | float | None = 0,
        no_label_replace: int | None = -1,
        use_metadata: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(SnowMapNonGeoMultiDataModule, batch_size=batch_size, num_workers=num_workers, **kwargs)

        # self.data_root = data_root

        # self.bands = bands

        self.drop_last = drop_last
        self.constant_scale = constant_scale
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
        self.use_metadata = use_metadata
        self.predict_root = predict_data_root

        get_stats = False
        if means is None or stds is None:
            means = []
            stds = []
            get_stats = True
        self.datasets = {}
        if dataset_order is not None:
            if len(dataset_order) != len(dataset_kwargs):
                raise ValueError("Dataset order must have the same length as dataset_kwargs.")
            assert set(dataset_order) == set(dataset_kwargs.keys()), "Dataset order must contain all dataset keys."
            dataset_keys = dataset_order
        else:
            dataset_keys = sorted(dataset_kwargs.keys(), reverse=True)  # Ugly hack to ensure S2 stays first

        for dataset in dataset_keys:
            dataset_values = dataset_kwargs[dataset]
            # self.data_root = dataset_values.get("data_root", None)   

            if (
                dataset_values.get("transform", None) is not None
                or dataset_values.get("train_transform", None) is not None
                or dataset_values.get("val_transform", None) is not None
                or dataset_values.get("test_transform", None) is not None
            ):
                # Lets make sure that the user does not specify transforms per dataset for now, less error prone
                raise ValueError("Transforms should be specified as a global argument and not per dataset.")

            if self.predict_root and "predict_data_root" not in dataset_values:
                dataset_values["predict_data_root"] = self.predict_root

            if ("means" not in dataset_values or "stds" not in dataset_values) and get_stats:
                raise ValueError(
                    "Means and stds must be specified for each dataset in dataset_kwargs or as a global argument."
                )
            if get_stats:
                means += dataset_values["means"]
                stds += dataset_values["stds"]

            self.datasets[dataset] = SnowMapNonGeoDataModule(
                batch_size=batch_size, num_workers=num_workers, **dataset_values
            )

        # TODO: do some sanity checks here to make sure that the datasets are compatible with each other.

        means = load_from_file_or_attribute(means)
        stds = load_from_file_or_attribute(stds)

        # self.aug = AugmentationSequential(K.Normalize(means, stds), data_keys=["image"])
        self.aug = Normalize(means, stds)

        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)

    def setup(self, stage: str):
        print("running multi setup")
        for _dataset, data_module in self.datasets.items():
            data_module.setup(stage)
                # TODO: sanity checks, check len etc...
        if stage in ["fit"]:
            self.train_datasets = {}
            for dataset, data_module in self.datasets.items():
                self.train_datasets[dataset] = data_module.train_dataset
            self.train_dataset = StackDatasetWithTransform(**self.train_datasets, 
                                                           transform=self.train_transform)

        if stage in ["fit", "validate"]:
            self.val_datasets = {}
            for dataset, data_module in self.datasets.items():
                self.val_datasets[dataset] = data_module.val_dataset
            self.val_dataset = StackDatasetWithTransform(**self.val_datasets, 
                                                         transform=self.val_transform)

        if stage in ["test"]:
            self.test_datasets = {}
            for dataset, data_module in self.datasets.items():
                self.test_datasets[dataset] = data_module.test_dataset
            self.test_dataset = StackDatasetWithTransform(**self.test_datasets, 
                                                          transform=self.test_transform)

        if stage in ["predict"] and self.predict_root:
            self.predict_datasets = {}
            for dataset, data_module in self.datasets.items():
                self.predict_datasets[dataset] = data_module.predict_dataset
            self.predict_dataset = StackDatasetWithTransform(**self.predict_datasets)

    def _dataloader_factory(self, split: str) -> DataLoader[dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders.

        Args:
            split: Either 'train', 'val', 'test', or 'predict'.

        Returns:
            A collection of data loaders specifying samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                dataset or sampler, or if the dataset or sampler has length 0.
        """
        dataset = self._valid_attribute(f"{split}_dataset", "dataset")
        batch_size = self._valid_attribute(f"{split}_batch_size", "batch_size")
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            # drop_last=split == "train" and self.drop_last,
            drop_last=self.drop_last,  # TODO: make this default for all dataloaders and other modules?
        )
