import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn


def rename_weight_dict(old_d, encoder):
    if encoder:
        for key in list(old_d.keys()):
            if "decoder" in key:
                old_d.pop(key)
            else:
                # name without the .encoder prefix
                new_key = key.split('.', 1)[1]
                # rename the key and delete the old one
                old_d[new_key] = old_d.pop(key)
        return old_d
    else:
        for key in list(old_d.keys()):
            if "encoder" in key:
                old_d.pop(key)
            else:
                # name without the .decoder prefix
                new_key = key.split('.', 1)[1]
                # rename the key and delete the old one
                old_d[new_key] = old_d.pop(key)
        return old_d


class ModelMaker:

    def __init__(self, *, model_params=None):
        self.model_name = model_params['model_name']
        self.num_classes = 1 # binary segmentation
        self.input_size = model_params['input_img_size']
        self.aug_type = model_params['aug_type']
        self.dropout = model_params['dropout']
        self.re_size = model_params['input_img_size']
        self.weights = model_params['weights']

    @staticmethod
    def set_parameter_requires_grad(model, feature_extracting) -> None:
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def make_model(self):
        if self.model_name == "mit_b1":
            from dlp.models.model_archs import SingleModelMiTB1
            model = SingleModelMiTB1(feature=95, out_channels=1, dropout=self.dropout)
            if self.weights == "trainval_lv2":
                weight_path = "./weights/single_mit_b1_lv2_trainval.pth"
                model.load_state_dict(torch.load(weight_path))
                return model
            model_preweight_path = "./weights/single_mit_b1_imagenet.pth"
            state_dict = torch.load(model_preweight_path)
            state_dict.pop("head.weight", None)
            state_dict.pop("head.bias", None)
            model.encoder.load_state_dict(state_dict)
            self.set_parameter_requires_grad(model.encoder, feature_extracting=True)
            print("the backbone imagenet weights were frozen. Decoder was not frozen")
            print("the weight path is ", model_preweight_path)
            return model
        elif self.model_name == "mit_b2":
            from dlp.models.model_archs import SingleModelMiTB2

            model = SingleModelMiTB2(feature=95, out_channels=1, dropout=self.dropout)
            if self.weights == "trainval_lv2":
                weight_path = "./weights/single_mit_b2_lv2_trainval.pth"
                model.load_state_dict(torch.load(weight_path))
                return model

            model_preweight_path = "./weights/single_mit_b2_imagenet.pth"
            state_dict = torch.load(model_preweight_path)
            state_dict.pop("head.weight", None)
            state_dict.pop("head.bias", None)
            model.encoder.load_state_dict(state_dict)
            self.set_parameter_requires_grad(model.encoder, feature_extracting=True)
            print("the backbone imagenet weights were frozen. Decoder was not frozen")
            return model
        elif self.model_name == "cgs_mit_b1":
            from dlp.models.model_archs import DualMiTB1CrossAtt
            model = DualMiTB1CrossAtt(heads=8, feature=95, out_channels=1, dropout=self.dropout)
            if self.weights == "trainval_single":
                # loading the best weights from the individual level 2 and 3 runs
                weight_path_detail = "./weights/single_mit_b1_lv2_trainval.pth"
                weight_path_context = "./weights/single_mit_b1_lv3_trainval.pth"
                # you can add an elif here and add your own weights
            elif self.weights == "trainval_cgs":
                weight_path = "./weights/cgs_mit_b1_trainval.pth"
                # model.load_state_dict(torch.load(weight_path)) # when running the standard code
                model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
                return model
            else:
                raise ValueError("incorrect dual model weight name was passed")
            new_context = rename_weight_dict(torch.load(weight_path_context), encoder=True)
            new_detail = rename_weight_dict(torch.load(weight_path_detail), encoder=True)

            model.encoder.encoder_detail.load_state_dict(new_detail)
            model.encoder.encoder_context.load_state_dict(new_context)
            self.set_parameter_requires_grad(model.encoder, feature_extracting=True)
            # now we load the weights for the decoder (lvl 2 only)
            detail_decoder = torch.load(weight_path_detail)
            detail_decoder_weights = rename_weight_dict(detail_decoder, encoder=False)
            load_result = model.decoder.lv2decoder.load_state_dict(detail_decoder_weights)
            assert len(load_result.missing_keys) == 0
            # all keys should have been used
            assert len(load_result.unexpected_keys) == 0
            # freezing the decoder
            self.set_parameter_requires_grad(model.decoder.lv2decoder, feature_extracting=True)
            # now we want to set the weights for MCA part of the model
            crosses = ["first_cross", "second_cross", "third_cross", "fourth_cross"]
            for cross in crosses:
                # https://stackoverflow.com/questions/44634972/how-to-access-a-field-of-a-namedtuple-using-a-variable-for-the-field-name
                # we want the context to be zero, this correpsonds to the queries
                getattr(model.decoder.mca, cross).queries.weight = nn.Parameter(
                    torch.zeros_like(getattr(model.decoder.mca, cross).queries.weight))
                # Identity matrix for weights
                getattr(model.decoder.mca, cross).keys.weight = nn.Parameter(
                    torch.eye(getattr(model.decoder.mca, cross).keys.weight.shape[0]))
                getattr(model.decoder.mca, cross).values.weight = nn.Parameter(
                    torch.eye(getattr(model.decoder.mca, cross).values.weight.shape[0]))
                getattr(model.decoder.mca, cross).projection.weight = nn.Parameter(
                    torch.eye(getattr(model.decoder.mca, cross).projection.weight.shape[0]))
                # set all the bias to zero
                getattr(model.decoder.mca, cross).queries.bias = nn.Parameter(
                    torch.zeros_like(getattr(model.decoder.mca, cross).queries.bias))
                getattr(model.decoder.mca, cross).keys.bias = nn.Parameter(
                    torch.zeros_like(getattr(model.decoder.mca, cross).keys.bias))
                getattr(model.decoder.mca, cross).values.bias = nn.Parameter(
                    torch.zeros_like(getattr(model.decoder.mca, cross).values.bias))
                getattr(model.decoder.mca, cross).projection.bias = nn.Parameter(
                    torch.zeros_like(getattr(model.decoder.mca, cross).projection.bias))

            # extra assert checks
            for param in model.encoder.parameters():
                assert param.requires_grad == False
            for param in model.decoder.lv2decoder.parameters():
                assert param.requires_grad == False
            for param in model.decoder.mca.parameters():
                assert param.requires_grad == True
            print(" the weights on the detail encoder are from: ", weight_path_detail)
            print(" the weights on the decoder are from: ", weight_path_detail)
            print(" the weights on the context encoder are from: ", weight_path_context)
            print(" the MCA was not frozen")
            print(" both encoder branches were frozen and decoder were frozen")
            return model
        elif self.model_name == "cgs_mit_b2":
            from dlp.models.model_archs import DualMiTB2CrossAtt
            model = DualMiTB2CrossAtt(heads=8, feature=95, out_channels=1, dropout=self.dropout)
            if self.weights == "trainval_single":
                # loading the best weights from the individual level 2 and 3 runs
                weight_path_detail = "./weights/single_mit_b2_lv2_trainval.pth"
                weight_path_context = "./weights/single_mit_b2_lv3_trainval.pth"
                # you can add an elif here and add your own weights
            elif self.weights == "trainval_cgs":
                weight_path = "./weights/cgs_mit_b2_trainval.pth"
                model.load_state_dict(torch.load(weight_path))
                return model
            else:
                raise ValueError("incorrect dual model weight name was passed")
            new_context = rename_weight_dict(torch.load(weight_path_context), encoder=True)
            new_detail = rename_weight_dict(torch.load(weight_path_detail), encoder=True)

            model.encoder.encoder_detail.load_state_dict(new_detail)
            model.encoder.encoder_context.load_state_dict(new_context)
            self.set_parameter_requires_grad(model.encoder, feature_extracting=True)
            # now we load the weights for the decoder (lvl 2 only)
            detail_decoder = torch.load(weight_path_detail)
            detail_decoder_weights = rename_weight_dict(detail_decoder, encoder=False)
            load_result = model.decoder.lv2decoder.load_state_dict(detail_decoder_weights)
            assert len(load_result.missing_keys) == 0
            # all keys should have been used
            assert len(load_result.unexpected_keys) == 0
            # freezing the decoder
            self.set_parameter_requires_grad(model.decoder.lv2decoder, feature_extracting=True)
            # now we want to set the weights for MCA part of the model
            crosses = ["first_cross", "second_cross", "third_cross", "fourth_cross"]
            for cross in crosses:
                # we want the context to be zero
                getattr(model.decoder.mca, cross).queries.weight = nn.Parameter(
                    torch.zeros_like(getattr(model.decoder.mca, cross).queries.weight))
                # Identity matrix for weights
                getattr(model.decoder.mca, cross).keys.weight = nn.Parameter(
                    torch.eye(getattr(model.decoder.mca, cross).keys.weight.shape[0]))
                getattr(model.decoder.mca, cross).values.weight = nn.Parameter(
                    torch.eye(getattr(model.decoder.mca, cross).values.weight.shape[0]))
                getattr(model.decoder.mca, cross).projection.weight = nn.Parameter(
                    torch.eye(getattr(model.decoder.mca, cross).projection.weight.shape[0]))
                # set all the bias to zero
                getattr(model.decoder.mca, cross).queries.bias = nn.Parameter(
                    torch.zeros_like(getattr(model.decoder.mca, cross).queries.bias))
                getattr(model.decoder.mca, cross).keys.bias = nn.Parameter(
                    torch.zeros_like(getattr(model.decoder.mca, cross).keys.bias))
                getattr(model.decoder.mca, cross).values.bias = nn.Parameter(
                    torch.zeros_like(getattr(model.decoder.mca, cross).values.bias))
                getattr(model.decoder.mca, cross).projection.bias = nn.Parameter(
                    torch.zeros_like(getattr(model.decoder.mca, cross).projection.bias))

            # extra assert checks
            for param in model.encoder.parameters():
                assert param.requires_grad == False
            for param in model.decoder.lv2decoder.parameters():
                assert param.requires_grad == False
            for param in model.decoder.mca.parameters():
                assert param.requires_grad == True
            print(" the weights on the detail encoder are from: ", weight_path_detail)
            print(" the weights on the decoder are from: ", weight_path_detail)
            print(" the weights on the context encoder are from: ", weight_path_context)
            print(" the MCA was not frozen")
            print(" both encoder branches were frozen and decoder were frozen")
            return model
        elif self.model_name == "swinv2_small":
            """
            The pretrained weights for the swin model are coming from timm 
            """
            from dlp.models.model_archs import SingleModelSwinSmall
            model = SingleModelSwinSmall(feature=85, out_channels=1, dropout=self.dropout)
            if self.weights == "trainval_lv2":
                weight_path = "./weights/single_swinv2_small_lv2_trainval.pth"
                model.load_state_dict(torch.load(weight_path))
                return model
            self.set_parameter_requires_grad(model.encoder, feature_extracting=True)
            print("the backbone imagenet weights were frozen. Decoder was not frozen")
            return model
        elif self.model_name == "swinv2_tiny":
            from dlp.models.model_archs import SingleModelSwinTiny
            model = SingleModelSwinTiny(feature=85, out_channels=1, dropout=self.dropout)
            if self.weights == "trainval_lv2":
                weight_path = "./weights/single_swinv2_tiny_lv2_trainval.pth"
                model.load_state_dict(torch.load(weight_path))
                return model

            self.set_parameter_requires_grad(model.encoder, feature_extracting=True)
            print("the backbone imagenet weights were frozen. Decoder was not frozen")
            return model
        elif self.model_name == "cgs_swinv2_tiny":
            from dlp.models.model_archs import DualSwinCrossAttTiny
            model = DualSwinCrossAttTiny(heads=8, feature=85, out_channels=1, dropout=self.dropout)

            if self.weights == "trainval_single":
                # loading the best weights from the individual level 2 and 3 runs
                weight_path_detail = "./weights/single_swinv2_tiny_lv2_trainval.pth.pth"
                weight_path_context = "./weights/single_swinv2_tiny_lv3_trainval.pth.pth"
                # you can add an elif here and add your own weights
            elif self.weights == "trainval_cgs":
                weight_path = "./weights/cgs_swinv2_tiny_trainval.pth"
                model.load_state_dict(torch.load(weight_path))
                return model
            else:
                raise ValueError("incorrect dual model weight name was passed")

            new_context = rename_weight_dict(torch.load(weight_path_context), encoder=True)
            new_detail = rename_weight_dict(torch.load(weight_path_detail), encoder=True)

            model.encoder.encoder_detail.load_state_dict(new_detail)
            model.encoder.encoder_context.load_state_dict(new_context)
            self.set_parameter_requires_grad(model.encoder, feature_extracting=True)
            # now we load the weights for the decoder (lvl 2 only)
            detail_decoder = torch.load(weight_path_detail)
            detail_decoder_weights = rename_weight_dict(detail_decoder, encoder=False)
            load_result = model.decoder.lv2decoder.load_state_dict(detail_decoder_weights)
            assert len(load_result.missing_keys) == 0
            # all keys should have been used
            assert len(load_result.unexpected_keys) == 0
            # freezing the decoder
            self.set_parameter_requires_grad(model.decoder.lv2decoder, feature_extracting=True)
            # now we want to set the weights for MCA part of the model
            crosses = ["first_cross", "second_cross", "third_cross", "fourth_cross"]
            for cross in crosses:
                # we want the context to be zero
                getattr(model.decoder.mca, cross).queries.weight = nn.Parameter(
                    torch.zeros_like(getattr(model.decoder.mca, cross).queries.weight))
                # Identity matrix for weights
                getattr(model.decoder.mca, cross).keys.weight = nn.Parameter(
                    torch.eye(getattr(model.decoder.mca, cross).keys.weight.shape[0]))
                getattr(model.decoder.mca, cross).values.weight = nn.Parameter(
                    torch.eye(getattr(model.decoder.mca, cross).values.weight.shape[0]))
                getattr(model.decoder.mca, cross).projection.weight = nn.Parameter(
                    torch.eye(getattr(model.decoder.mca, cross).projection.weight.shape[0]))
                # set all the bias to zero
                getattr(model.decoder.mca, cross).queries.bias = nn.Parameter(
                    torch.zeros_like(getattr(model.decoder.mca, cross).queries.bias))
                getattr(model.decoder.mca, cross).keys.bias = nn.Parameter(
                    torch.zeros_like(getattr(model.decoder.mca, cross).keys.bias))
                getattr(model.decoder.mca, cross).values.bias = nn.Parameter(
                    torch.zeros_like(getattr(model.decoder.mca, cross).values.bias))
                getattr(model.decoder.mca, cross).projection.bias = nn.Parameter(
                    torch.zeros_like(getattr(model.decoder.mca, cross).projection.bias))

            # extra assert checks
            for param in model.encoder.parameters():
                assert param.requires_grad == False
            for param in model.decoder.lv2decoder.parameters():
                assert param.requires_grad == False
            for param in model.decoder.mca.parameters():
                assert param.requires_grad == True
            print(" the weights on the detail encoder are from: ", weight_path_detail)
            print(" the weights on the decoder are from: ", weight_path_detail)
            print(" the weights on the context encoder are from: ", weight_path_context)
            print(" the MCA was not frozen")
            print(" both encoder branches were frozen and decoder were frozen")
            return model

        elif self.model_name == "cgs_swinv2_small":
            from dlp.models.model_archs import DualSwinCrossAttSmall
            model = DualSwinCrossAttSmall(heads=8, feature=85, out_channels=1, dropout=self.dropout)
            if self.weights == "trainval_single":
                # loading the best weights from the individual level 2 and 3 runs
                weight_path_detail = "./weights/single_swinv2_small_lv2_trainval.pth.pth"
                weight_path_context = "./weights/single_swinv2_small_lv3_trainval.pth.pth"
                # you can add an elif here and add your own weights
            elif self.weights == "trainval_cgs":
                weight_path = "./weights/cgs_swinv2_small_trainval.pth"
                model.load_state_dict(torch.load(weight_path))
                return model
            else:
                raise ValueError("incorrect dual model weight name was passed")

            new_context = rename_weight_dict(torch.load(weight_path_context), encoder=True)
            new_detail = rename_weight_dict(torch.load(weight_path_detail), encoder=True)

            model.encoder.encoder_detail.load_state_dict(new_detail)
            model.encoder.encoder_context.load_state_dict(new_context)
            self.set_parameter_requires_grad(model.encoder, feature_extracting=True)
            # now we load the weights for the decoder (lvl 2 only)
            detail_decoder = torch.load(weight_path_detail)
            detail_decoder_weights = rename_weight_dict(detail_decoder, encoder=False)
            load_result = model.decoder.lv2decoder.load_state_dict(detail_decoder_weights)
            assert len(load_result.missing_keys) == 0
            # all keys should have been used
            assert len(load_result.unexpected_keys) == 0
            # freezing the decoder
            self.set_parameter_requires_grad(model.decoder.lv2decoder, feature_extracting=True)
            # now we want to set the weights for MCA part of the model
            crosses = ["first_cross", "second_cross", "third_cross", "fourth_cross"]
            for cross in crosses:
                # we want the context to be zero
                getattr(model.decoder.mca, cross).queries.weight = nn.Parameter(
                    torch.zeros_like(getattr(model.decoder.mca, cross).queries.weight))
                # Identity matrix for weights
                getattr(model.decoder.mca, cross).keys.weight = nn.Parameter(
                    torch.eye(getattr(model.decoder.mca, cross).keys.weight.shape[0]))
                getattr(model.decoder.mca, cross).values.weight = nn.Parameter(
                    torch.eye(getattr(model.decoder.mca, cross).values.weight.shape[0]))
                getattr(model.decoder.mca, cross).projection.weight = nn.Parameter(
                    torch.eye(getattr(model.decoder.mca, cross).projection.weight.shape[0]))
                # set all the bias to zero
                getattr(model.decoder.mca, cross).queries.bias = nn.Parameter(
                    torch.zeros_like(getattr(model.decoder.mca, cross).queries.bias))
                getattr(model.decoder.mca, cross).keys.bias = nn.Parameter(
                    torch.zeros_like(getattr(model.decoder.mca, cross).keys.bias))
                getattr(model.decoder.mca, cross).values.bias = nn.Parameter(
                    torch.zeros_like(getattr(model.decoder.mca, cross).values.bias))
                getattr(model.decoder.mca, cross).projection.bias = nn.Parameter(
                    torch.zeros_like(getattr(model.decoder.mca, cross).projection.bias))

            # extra assert checks
            for param in model.encoder.parameters():
                assert param.requires_grad == False
            for param in model.decoder.lv2decoder.parameters():
                assert param.requires_grad == False
            for param in model.decoder.mca.parameters():
                assert param.requires_grad == True
            print(" the weights on the detail encoder are from: ", weight_path_detail)
            print(" the weights on the decoder are from: ", weight_path_detail)
            print(" the weights on the context encoder are from: ", weight_path_context)
            print(" the MCA was not frozen")
            print("both encoder branches were frozen and decoder was frozen")
            return model
        else:
            raise ValueError("incorrect model name was passed")

    @staticmethod
    def print_model_params(model) -> None:
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print("the total number of parameters in the model is ", pytorch_total_params)
        # same idea but now only for trainable parameters
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("the total number of TRAINABLE parameters in the model is ", pytorch_total_params)
        # encoder
        pytorch_total_params = sum(p.numel() for p in model.encoder.parameters())
        print("the total number of parameters in the encoder is ", pytorch_total_params)
        # decoder
        pytorch_total_params = sum(p.numel() for p in model.decoder.parameters())
        print("the total number of parameters in the decoder is ", pytorch_total_params)

    def make_train_transform(self):
        if self.aug_type == 'stand':
            train_transform = A.Compose(
                [
                    A.Resize(height=self.re_size, width=self.re_size, always_apply=True),
                    # A.CenterCrop(p=1, height=self.input_size, width=self.input_size),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255.0,
                    ),
                    ToTensorV2(),
                ], is_check_shapes=False,
            )

        elif self.aug_type == 'aggresive':
            train_transform = A.Compose(
                [
                    A.Resize(height=self.re_size, width=self.re_size, always_apply=True),
                    # A.CenterCrop(p=1, height=self.input_size, width=self.input_size),
                    A.CoarseDropout(max_holes=5, max_height=16, max_width=16, mask_fill_value=None, p=0.25),
                    A.GaussianBlur(blur_limit=(3, 3), p=0.5),
                    A.Rotate(limit=45, p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Affine(scale=1.0, translate_percent=0.125, shear=None, p=0.5),
                    A.Affine(scale=(.5, 1.5), p=0.25),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255.0,
                    ),
                    ToTensorV2(),
                ], is_check_shapes=False,
            )

        elif self.aug_type == 'moreagg':
            train_transform = A.Compose(
                [
                    A.Resize(height=self.re_size, width=self.re_size, always_apply=True),
                    A.Transpose(p=0.5),
                    # A.CenterCrop(p=1, height=self.input_size, width=self.input_size),
                    A.CoarseDropout(max_holes=10, max_height=16, max_width=16, mask_fill_value=None, p=0.5),
                    A.GaussianBlur(blur_limit=(3, 3), p=0.5),
                    A.Rotate(limit=45, p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.HueSaturationValue(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Affine(scale=1.0, translate_percent=0.125, shear=None, p=0.5),
                    A.Affine(scale=(.5, 1.5), p=0.5),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255.0,
                    ),
                    ToTensorV2(),
                ], is_check_shapes=False,
            )
        elif self.aug_type == 'evenmoreagg':
            train_transform = A.Compose(
                [
                    A.Resize(height=self.re_size, width=self.re_size, always_apply=True),
                    A.Transpose(p=0.5),
                    # A.CenterCrop(p=1, height=self.input_size, width=self.input_size),
                    A.CoarseDropout(max_holes=10, max_height=16, max_width=16, mask_fill_value=None, p=0.5),
                    A.Solarize(p=0.5),
                    A.GaussianBlur(blur_limit=(3, 3), p=0.5),
                    A.Rotate(limit=45, p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.HueSaturationValue(p=0.5),
                    A.GaussNoise(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Affine(scale=1.0, translate_percent=0.125, shear=None, p=0.5),
                    A.Affine(scale=(.5, 1.5), p=0.5),
                    A.ColorJitter(p=0.5),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255.0,
                    ),
                    ToTensorV2(),
                ], is_check_shapes=False,
            )
        elif self.aug_type == 'mostagg':
            train_transform = A.Compose(
                [
                    A.Resize(height=self.re_size, width=self.re_size, always_apply=True),
                    A.Transpose(p=0.75),
                    # A.CenterCrop(p=1, height=self.input_size, width=self.input_size),
                    A.CoarseDropout(max_holes=10, max_height=16, max_width=16, mask_fill_value=None, p=0.75),
                    A.Solarize(p=0.75),
                    A.GaussianBlur(blur_limit=(3, 3), p=0.75),
                    A.Rotate(limit=45, p=0.75),
                    A.HorizontalFlip(p=0.75),
                    A.HueSaturationValue(p=0.75),
                    A.GaussNoise(p=0.75),
                    A.VerticalFlip(p=0.75),
                    A.Affine(scale=1.0, translate_percent=0.125, shear=None, p=0.75),
                    A.Affine(scale=(.5, 1.5), p=0.75),
                    A.ColorJitter(p=0.75),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255.0,
                    ),
                    ToTensorV2(),
                ], is_check_shapes=False,
            )
        elif self.aug_type == 'some':
            train_transform = A.Compose(
                [
                    A.Resize(height=self.re_size, width=self.re_size, always_apply=True),
                    # A.CenterCrop(p=1, height=self.input_size, width=self.input_size),
                    # A.GaussianBlur(blur_limit=(3, 3), p=0.25),
                    A.Rotate(limit=45, p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    # A.Affine(scale=1.0, translate_percent=0.125, shear=(-20, 20), p=0.5),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255.0,
                    ),
                    ToTensorV2(),
                ], is_check_shapes=False,
            )
        else:
            raise ValueError("incorrect aug type was passed")

        print("the train transform is ")
        print(train_transform)
        return train_transform

    def make_val_transform(self):
        """
        albumentations expects the image in H X W X C the final operation ToTensorV2() puts the image in the following
        order C x H x W which is what pytorch expects.
        """
        val_transform = A.Compose(
            [
                A.Resize(height=self.input_size, width=self.input_size, always_apply=True),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ], is_check_shapes=False,
        )
        print("the val transform is ")
        print(val_transform)
        return val_transform
