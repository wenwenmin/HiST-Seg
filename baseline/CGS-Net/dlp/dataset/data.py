from torch.utils.data import DataLoader
import os
from .data_classes import DataSegmentCamelyon16, DualDataSegmentCamelyon16


class DataMaker:
    def __init__(self, *, model_params: dict):

        self.data_dir = model_params['data_dir']
        self.model_type = model_params['model_type']
        self.data_name = model_params['data_name']
        if self.model_type == "segment" and self.data_name == "c16_lv2":
            self.test_path_image = os.path.join(self.data_dir, "test_images_lv2/")
            self.test_path_mask = os.path.join(self.data_dir, "test_masks_lv2/")
            self.val_path_image = os.path.join(self.data_dir, "val_images_lv2/")
            self.val_path_mask = os.path.join(self.data_dir, "val_masks_lv2/")
            self.train_path_image = os.path.join(self.data_dir, "train_images_lv2/")
            self.train_path_mask = os.path.join(self.data_dir, "train_masks_lv2/")
            self.trainval_path_image = os.path.join(self.data_dir, "trainval_images_lv2/")
            self.trainval_path_mask = os.path.join(self.data_dir, "trainval_masks_lv2/")
        elif self.model_type == "segment" and self.data_name == "c16_lv3":
            self.test_path_image = os.path.join(self.data_dir, "test_images_lv3/")
            self.test_path_mask = os.path.join(self.data_dir, "test_masks_lv3/")
            self.val_path_image = os.path.join(self.data_dir, "val_images_lv3/")
            self.val_path_mask = os.path.join(self.data_dir, "val_masks_lv3/")
            self.train_path_image = os.path.join(self.data_dir, "train_images_lv3/")
            self.train_path_mask = os.path.join(self.data_dir, "train_masks_lv3/")
            self.trainval_path_image = os.path.join(self.data_dir, "trainval_images_lv3/")
            self.trainval_path_mask = os.path.join(self.data_dir, "trainval_masks_lv3/")
        elif self.model_type == "segmentdual" and self.data_name == "c16":
            # the detail paths
            self.test_path_image = os.path.join(self.data_dir, "test_images_lv2/")
            self.test_path_mask = os.path.join(self.data_dir, "test_masks_lv2/")
            self.val_path_image = os.path.join(self.data_dir, "val_images_lv2/")
            self.val_path_mask = os.path.join(self.data_dir, "val_masks_lv2/")
            self.train_path_image = os.path.join(self.data_dir, "train_images_lv2/")
            self.train_path_mask = os.path.join(self.data_dir, "train_masks_lv2/")
            self.trainval_path_image = os.path.join(self.data_dir, "trainval_images_lv2/")
            self.trainval_path_mask = os.path.join(self.data_dir, "trainval_masks_lv2/")
            # these are for the context paths
            self.test_path_image_context = os.path.join(self.data_dir, "test_images_lv3/")
            self.test_path_mask_context = os.path.join(self.data_dir, "test_masks_lv3/")
            self.val_path_image_context = os.path.join(self.data_dir, "val_images_lv3/")
            self.val_path_mask_context = os.path.join(self.data_dir, "val_masks_lv3/")
            self.train_path_image_context = os.path.join(self.data_dir, "train_images_lv3/")
            self.train_path_mask_context = os.path.join(self.data_dir, "train_masks_lv3/")
            self.trainval_path_image_context = os.path.join(self.data_dir, "trainval_images_lv3/")
            self.trainval_path_mask_context = os.path.join(self.data_dir, "trainval_masks_lv3/")
        else:
            raise Exception("Wrong model type passed")

    def get_loaders_segment(self, train_transform, val_transform, test_transform, model_params, trainval: bool):
        if not trainval:
            train_ds = DataSegmentCamelyon16(
                image_dir=self.train_path_image,
                mask_dir=self.train_path_mask,
                transform=train_transform,
            )

            train_loader = DataLoader(
                train_ds,
                batch_size=model_params['batch_size'],
                num_workers=model_params['num_workers'],
                pin_memory=model_params['pin_mem'],
                shuffle=True,
                drop_last=True,
            )
        elif trainval:
            # this is the combined dataset of training and validations
            # this is  when we already picked the best model using the val loss and now
            # are retraining with the combined dataset until we hit the epoch that yielded the lowest
            # val loss.
            train_ds = DataSegmentCamelyon16(
                image_dir=self.trainval_path_image,
                mask_dir=self.trainval_path_mask,
                transform=train_transform,
            )

            train_loader = DataLoader(
                train_ds,
                batch_size=model_params['batch_size'],
                num_workers=model_params['num_workers'],
                pin_memory=model_params['pin_mem'],
                shuffle=True,
                drop_last=True,
            )
        else:
            raise Exception("wrong testing value passed")

        val_ds = DataSegmentCamelyon16(
            image_dir=self.val_path_image,
            mask_dir=self.val_path_mask,
            transform=val_transform,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=model_params['batch_size'],
            num_workers=model_params['num_workers'],
            pin_memory=model_params['pin_mem'],
            shuffle=False,
            drop_last=True,
        )

        test_ds = DataSegmentCamelyon16(
            image_dir=self.test_path_image,
            mask_dir=self.test_path_mask,
            transform=test_transform,
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=model_params['batch_size'],
            num_workers=model_params['num_workers'],
            pin_memory=model_params['pin_mem'],
            shuffle=False,
            drop_last=True,
        )

        return train_loader, val_loader, test_loader

    def get_loaders_segmentdual(self, train_transform, val_transform, test_transform, model_params, trainval):
        """
        trainval: boolean if True using combined Train and val dataset
        if false train and val are separate datasets
        """
        if not trainval:
            train_ds = DualDataSegmentCamelyon16(
                image_dir_detail=self.train_path_image,
                image_dir_context=self.train_path_image_context,
                mask_dir_detail=self.train_path_mask,
                transform=train_transform,
            )

            train_loader = DataLoader(
                train_ds,
                batch_size=model_params['batch_size'],
                num_workers=model_params['num_workers'],
                pin_memory=model_params['pin_mem'],
                shuffle=True,
                drop_last=True,
            )
        elif trainval:
            train_ds = DualDataSegmentCamelyon16(
                image_dir_detail=self.trainval_path_image,
                image_dir_context=self.trainval_path_image_context,
                mask_dir_detail=self.trainval_path_mask,
                transform=train_transform,
            )

            train_loader = DataLoader(
                train_ds,
                batch_size=model_params['batch_size'],
                num_workers=model_params['num_workers'],
                pin_memory=model_params['pin_mem'],
                shuffle=True,
                drop_last=True,
            )
        else:
            raise Exception("wrong testing value passed")

        val_ds = DualDataSegmentCamelyon16(
            image_dir_detail=self.val_path_image,
            image_dir_context=self.val_path_image_context,
            mask_dir_detail=self.val_path_mask,
            transform=val_transform,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=model_params['batch_size'],
            num_workers=model_params['num_workers'],
            pin_memory=model_params['pin_mem'],
            shuffle=False,
            drop_last=True,
        )

        test_ds = DualDataSegmentCamelyon16(
            image_dir_detail=self.test_path_image,
            image_dir_context=self.test_path_image_context,
            mask_dir_detail=self.test_path_mask,
            transform=test_transform,
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=model_params['batch_size'],
            num_workers=model_params['num_workers'],
            pin_memory=model_params['pin_mem'],
            shuffle=False,
            drop_last=True,
        )

        return train_loader, val_loader, test_loader

    def get_loaders(self, model_params, train_transform, val_transform, test_transform, trainval=False) -> tuple:
        if not trainval:
            # this is the standard training loop. I return the test loader but it is not used in the training phase
            if model_params['model_type'] == "segment":
                train_loader, val_loader, test_loader = self.get_loaders_segment(train_transform=train_transform,
                                                                                 val_transform=val_transform,
                                                                                 test_transform=test_transform,
                                                                                 model_params=model_params,
                                                                                 trainval=trainval)
            elif model_params['model_type'] == "segmentdual":
                train_loader, val_loader, test_loader = self.get_loaders_segmentdual(train_transform=train_transform,
                                                                                     val_transform=val_transform,
                                                                                     test_transform=test_transform,
                                                                                     model_params=model_params,
                                                                                     trainval=trainval)

            else:
                raise Exception("wrong model_type passed")

            return train_loader, val_loader, test_loader
        elif trainval:
            if model_params['model_type'] == "segment":
                train_loader, _, test_loader = self.get_loaders_segment(train_transform=train_transform,
                                                                        val_transform=val_transform,
                                                                        test_transform=test_transform,
                                                                        model_params=model_params, trainval=trainval)
            elif model_params['model_type'] == "segmentdual":
                train_loader, _, test_loader = self.get_loaders_segmentdual(train_transform=train_transform,
                                                                            val_transform=val_transform,
                                                                            test_transform=test_transform,
                                                                            model_params=model_params,
                                                                            trainval=trainval)

            else:
                raise Exception("wrong model_type passed")
            return train_loader, _, test_loader
