import torch
import torch.nn as nn

from dlp.models.encoders import DualEncoderMiTB1, DualEncoderMiTB2, SwinEncoderDualSmall, SwinEncoderTiny, \
    SwinEncoderDualTiny
from dlp.models.decoders import DualDecoderMiT, DualDecoderSwin
from dlp.models.encoders import MixVisionTransformer, SwinEncoderSmall
from dlp.models.decoders import UnetDecoderSingleMiTB1, UnetDecoderSingleSwin
from functools import partial


# SWIN Models
class SingleModelSwinSmall(torch.nn.Module):
    def __init__(self, feature, out_channels, dropout):
        super(SingleModelSwinSmall, self).__init__()
        self.encoder = SwinEncoderSmall(imagenet_pretrained=True)
        self.decoder = UnetDecoderSingleSwin(feature, out_channels, dropout)

    def forward(self, x):
        x00, x10, x20, x30 = self.encoder(x)
        return self.decoder(x00=x00, x10=x10, x20=x20, x30=x30)


class SingleModelSwinTiny(torch.nn.Module):
    def __init__(self, feature, out_channels, dropout):
        super(SingleModelSwinTiny, self).__init__()
        self.encoder = SwinEncoderTiny(imagenet_pretrained=True)
        self.decoder = UnetDecoderSingleSwin(feature, out_channels, dropout)

    def forward(self, x):
        x00, x10, x20, x30 = self.encoder(x)
        return self.decoder(x00=x00, x10=x10, x20=x20, x30=x30)


class DualSwinCrossAttSmall(torch.nn.Module):
    def __init__(self, heads, feature, out_channels, dropout):
        super(DualSwinCrossAttSmall, self).__init__()
        self.encoder = SwinEncoderDualSmall()
        self.decoder = DualDecoderSwin(heads, feature, out_channels, dropout)

    def forward(self, x_detail, x_context):
        x00_detail, x00_context, x10_detail, x10_context, x20_detail, x20_context, x30_detail, x30_context = self.encoder(
            x_detail, x_context)
        return self.decoder(x00_detail=x00_detail, x00_context=x00_context, x10_detail=x10_detail,
                            x10_context=x10_context, x20_detail=x20_detail, x20_context=x20_context,
                            x30_detail=x30_detail, x30_context=x30_context)


class DualSwinCrossAttTiny(torch.nn.Module):
    def __init__(self, heads, feature, out_channels, dropout):
        super(DualSwinCrossAttTiny, self).__init__()
        self.encoder = SwinEncoderDualTiny()
        self.decoder = DualDecoderSwin(heads, feature, out_channels, dropout)

    def forward(self, x_detail, x_context):
        x00_detail, x00_context, x10_detail, x10_context, x20_detail, x20_context, x30_detail, x30_context = self.encoder(
            x_detail, x_context)
        return self.decoder(x00_detail=x00_detail, x00_context=x00_context, x10_detail=x10_detail,
                            x10_context=x10_context, x20_detail=x20_detail, x20_context=x20_context,
                            x30_detail=x30_detail, x30_context=x30_context)


# MiT Models
class DualMiTB2CrossAtt(torch.nn.Module):
    def __init__(self, heads, feature, out_channels, dropout):
        super(DualMiTB2CrossAtt, self).__init__()
        self.encoder = DualEncoderMiTB2()
        self.decoder = DualDecoderMiT(heads, feature, out_channels, dropout)

    def forward(self, x_detail, x_context):
        x00_detail, x00_context, x10_detail, x10_context, x20_detail, x20_context, x30_detail, x30_context = self.encoder(
            x_detail, x_context)
        return self.decoder(x00_detail=x00_detail, x00_context=x00_context, x10_detail=x10_detail,
                            x10_context=x10_context, x20_detail=x20_detail, x20_context=x20_context,
                            x30_detail=x30_detail, x30_context=x30_context)


class DualMiTB1CrossAtt(torch.nn.Module):
    def __init__(self, heads, feature, out_channels, dropout):
        super(DualMiTB1CrossAtt, self).__init__()
        self.encoder = DualEncoderMiTB1()
        self.decoder = DualDecoderMiT(heads, feature, out_channels, dropout)

    def forward(self, x_detail, x_context):
        x00_detail, x00_context, x10_detail, x10_context, x20_detail, x20_context, x30_detail, x30_context = self.encoder(
            x_detail, x_context)
        return self.decoder(x00_detail=x00_detail, x00_context=x00_context, x10_detail=x10_detail,
                            x10_context=x10_context, x20_detail=x20_detail, x20_context=x20_context,
                            x30_detail=x30_detail, x30_context=x30_context)


class SingleModelMiTB1(torch.nn.Module):
    def __init__(self, feature, out_channels, dropout):
        super(SingleModelMiTB1, self).__init__()
        self.encoder = MixVisionTransformer(patch_size=4,
                                            embed_dims=[64, 128, 320, 512],
                                            num_heads=[1, 2, 5, 8],
                                            mlp_ratios=[4, 4, 4, 4],
                                            qkv_bias=True,
                                            norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                            depths=[2, 2, 2, 2],
                                            sr_ratios=[8, 4, 2, 1],
                                            drop_rate=0.0,
                                            drop_path_rate=0.1, )
        self.decoder = UnetDecoderSingleMiTB1(feature, out_channels, dropout)

    def forward(self, x):
        x00, x10, x20, x30 = self.encoder(x)
        return self.decoder(x00=x00, x10=x10, x20=x20, x30=x30)


class SingleModelMiTB2(torch.nn.Module):
    def __init__(self, feature, out_channels, dropout):
        super(SingleModelMiTB2, self).__init__()
        self.encoder = MixVisionTransformer(patch_size=4,
                                            embed_dims=[64, 128, 320, 512],
                                            num_heads=[1, 2, 5, 8],
                                            mlp_ratios=[4, 4, 4, 4],
                                            qkv_bias=True,
                                            norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                            depths=[3, 4, 6, 3],
                                            sr_ratios=[8, 4, 2, 1],
                                            drop_rate=0.0,
                                            drop_path_rate=0.1, )
        self.decoder = UnetDecoderSingleMiTB1(feature, out_channels, dropout)

    def forward(self, x):
        x00, x10, x20, x30 = self.encoder(x)
        return self.decoder(x00=x00, x10=x10, x20=x20, x30=x30)


def gen_dual_b1(weights=None):
    model = DualMiTB1CrossAtt(heads=8, feature=95, out_channels=1, dropout=0)
    if weights:
        path = './weights/cgs_mit_b1_trainval.pth'
        print("the weights are found in ", path)
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        return model
    # the code below is for setting the MCA modules. if we pass weights then this is not needed.
    crosses = ["first_cross", "second_cross", "third_cross", "fourth_cross"]
    for cross in crosses:
        # https://stackoverflow.com/questions/44634972/how-to-access-a-field-of-a-namedtuple-using-a-variable-for-the-field-name
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
    return model


def look_at_mca_wegihts(model, wtype='queries'):
    crosses = ["first_cross", "second_cross", "third_cross", "fourth_cross"]
    for cross in crosses:
        print('--------------', cross, '-----------------------')
        cross_wegihts = getattr(getattr(model.decoder.mca, cross), wtype).weight
        average = torch.mean(cross_wegihts)
        median = torch.median(cross_wegihts)
        std_dev = torch.std(cross_wegihts)

        print("Average:", average.item())
        print("Median:", median.item())
        print("Standard Deviation:", std_dev.item())


if __name__ == "__main__":
    dual_model_og = gen_dual_b1()
    dual_model_weights = gen_dual_b1(weights=True)
    print('MiT B1 Dual model model')
    look_at_mca_wegihts(dual_model_og)
    print('weights model')
    print('looking at values')
    look_at_mca_wegihts(dual_model_weights, wtype='values')
    print('looking at keys')
    look_at_mca_wegihts(dual_model_weights, wtype='keys')
    print('looking at queries')
    look_at_mca_wegihts(dual_model_weights, wtype='queries')
    print('done')
