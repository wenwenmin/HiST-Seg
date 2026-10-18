from typing import Tuple
import torch.nn as nn
from torch import Tensor
from einops import rearrange
from einops.layers.torch import Rearrange
import torch
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, drop_out):
        super(DoubleConv, self).__init__()
        self.dc_kernel_size = 3
        self.dc_stride = 1
        self.dc_padding = 1  # this means that the input and the output will have the same size "same convolution"

        self.conv = nn.Sequential(
            # the bias is False since we are using Batchnorm
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=(self.dc_kernel_size, self.dc_kernel_size),
                      stride=(self.dc_stride, self.dc_stride), padding=(self.dc_padding, self.dc_padding),
                      bias=False),
            nn.BatchNorm2d(out_channels),
            # go to the bottom of the following link to see what inplace does, basically it saves some memory alloc
            # by "overwritting" the og input https://discuss.pytorch.org/t/whats-the-difference-between-nn-relu-and-nn-relu-inplace-true/948/10
            nn.ReLU(inplace=False),
            nn.Dropout2d(p=drop_out, inplace=False),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=(self.dc_kernel_size, self.dc_kernel_size),
                      stride=(self.dc_stride, self.dc_stride), padding=(self.dc_padding, self.dc_padding),
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Dropout2d(p=drop_out, inplace=False),
        )

    def forward(self, x):
        return self.conv(x)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, emb_size: int = 1024, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)

        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        # self.scaling = 1
        self.scaling = (self.emb_size // num_heads) ** -0.5

    def forward(self, x_detail: Tensor, x_context: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x_context), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x_detail), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x_detail), "b n (h d) -> b h n d", h=self.num_heads)
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        att = F.softmax(energy * self.scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 1024, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)

        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        self.scaling = (self.emb_size // num_heads) ** -0.5

    def forward(self, input: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(input), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(input), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(input), "b n (h d) -> b h n d", h=self.num_heads)
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        att = F.softmax(energy * self.scaling, dim=-1)
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class MCAttention(torch.nn.Module):
    def __init__(self, heads, dropout, emb_sizes) -> None:
        super().__init__()
        self.patch_flatten_first = Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=2, s2=2)
        self.patch_flatten_second = Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=1, s2=1)
        self.patch_flatten_third = Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=1, s2=1)
        self.patch_flatten_fourth = Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=1, s2=1)

        self.make_2d_first = Rearrange('b (h w) (s1 s2 c) -> b c (h s1) (w s2)', h=28, w=28, s1=2, s2=2)
        self.make_2d_second = Rearrange('b (h w) (s1 s2 c) -> b c (h s1) (w s2)', h=28, w=28, s1=1, s2=1)
        self.make_2d_third = Rearrange('b (h w) (s1 s2 c) -> b c (h s1) (w s2)', h=14, w=14, s1=1, s2=1)
        self.make_2d_fourth = Rearrange('b (h w) (s1 s2 c) -> b c (h s1) (w s2)', h=7, w=7, s1=1, s2=1)

        self.first_cross = MultiHeadCrossAttention(emb_size=emb_sizes[0], num_heads=heads, dropout=dropout)
        self.second_cross = MultiHeadCrossAttention(emb_size=emb_sizes[1], num_heads=heads, dropout=dropout)
        self.third_cross = MultiHeadCrossAttention(emb_size=emb_sizes[2], num_heads=heads, dropout=dropout)
        self.fourth_cross = MultiHeadCrossAttention(emb_size=emb_sizes[3], num_heads=heads, dropout=dropout)

    def forward(self, x00_detail, x00_context, x10_detail, x10_context, x20_detail, x20_context, x30_detail,
                x30_context) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # First cross (x0 cross)
        x00_detail_1d = self.patch_flatten_first(x00_detail)
        x00_context_1d = self.patch_flatten_first(x00_context)

        x0_cross_output = self.first_cross(x_detail=x00_detail_1d, x_context=x00_context_1d)
        x0_cross = self.make_2d_first(x0_cross_output)
        x00_detail = x00_detail + x0_cross

        # Second cross (x1 cross)
        x10_detail_1d = self.patch_flatten_second(x10_detail)
        x10_context_1d = self.patch_flatten_second(x10_context)
        x1_cross_output = self.second_cross(x_detail=x10_detail_1d, x_context=x10_context_1d)
        x1_cross = self.make_2d_second(x1_cross_output)
        x10_detail = x10_detail + x1_cross
        # Third cross (x2 cross)
        x20_detail_1d = self.patch_flatten_third(x20_detail)
        x20_context_1d = self.patch_flatten_third(x20_context)
        x2_cross_ouput = self.third_cross(x_detail=x20_detail_1d, x_context=x20_context_1d)
        x2_cross = self.make_2d_third(x2_cross_ouput)
        x20_detail = x20_detail + x2_cross

        # fourth cross (x3 cross)
        x30_detail_1d = self.patch_flatten_fourth(x30_detail)
        x30_context_1d = self.patch_flatten_fourth(x30_context)
        x3_cross_output = self.fourth_cross(x_detail=x30_detail_1d, x_context=x30_context_1d)
        x3_cross = self.make_2d_fourth(x3_cross_output)
        x30_detail = x30_detail + x3_cross
        return x00_detail, x10_detail, x20_detail, x30_detail


class UnetDecoderCross(torch.nn.Module):
    def __init__(self, feature, out_channels, dropout, emb_sizes) -> None:
        """
        for MiT emb_sizes = (512,320,128,64)
        for swin emb_sizes = (768,384,192,96)

        """
        super().__init__()

        self.x31_upconv = nn.ConvTranspose2d(feature, feature, kernel_size=(2, 2), stride=(2, 2))
        self.x31_doubleconv = DoubleConv(emb_sizes[0], feature, drop_out=dropout)
        self.x22_upconv = nn.ConvTranspose2d(feature, feature, kernel_size=(2, 2), stride=(2, 2))
        self.x22_doubleconv = DoubleConv(emb_sizes[1] + feature, feature, drop_out=dropout)
        self.x13_upconv = nn.ConvTranspose2d(feature, feature, kernel_size=(2, 2), stride=(2, 2))
        self.x13_doubleconv = DoubleConv(emb_sizes[2] + feature, feature, drop_out=dropout)
        self.x04_upconv = nn.ConvTranspose2d(feature, feature, kernel_size=(2, 2), stride=(2, 2))
        self.x04_doubleconv = DoubleConv(emb_sizes[3] + feature, feature, drop_out=dropout)

        # extra decoder layers
        self.intermedfinal_upconv = nn.ConvTranspose2d(feature, feature, kernel_size=(2, 2), stride=(2, 2))
        self.intermedfinal_doubleconv = DoubleConv(feature, feature, drop_out=dropout)
        self.final_upconv = nn.ConvTranspose2d(feature, feature, kernel_size=(2, 2), stride=(2, 2))

        self.final_doubleconv = DoubleConv(feature, feature, drop_out=dropout)
        self.segmentation_head = nn.Conv2d(feature, out_channels, kernel_size=(1, 1))  # segmentation output layer

    def forward(self, x00_detail, x10_detail, x20_detail, x30_detail) -> torch.Tensor:
        x31 = self.x31_doubleconv(x30_detail)
        x31 = self.x31_upconv(x31)  # bottlekneck hence a little different
        x22 = torch.cat((x20_detail, x31), dim=1)
        x22 = self.x22_doubleconv(x22)

        x13 = self.x13_upconv(x22)
        x13 = torch.cat((x10_detail, x13), dim=1)
        x13 = self.x13_doubleconv(x13)

        x04 = self.x04_upconv(x13)
        x04 = torch.cat((x00_detail, x04), dim=1)
        x04 = self.x04_doubleconv(x04)
        # if end the model here then Segformer output dimensions
        x0interfinal = self.intermedfinal_upconv(x04)
        x0interfinal = self.intermedfinal_doubleconv(x0interfinal)

        x0final = self.final_upconv(x0interfinal)
        x0final = self.final_doubleconv(x0final)
        return self.segmentation_head(x0final)


class DualDecoderMiT(torch.nn.Module):
    def __init__(self, heads, feature, out_channels, dropout):
        super().__init__()
        self.lv2decoder = UnetDecoderCross(feature=feature, out_channels=out_channels, dropout=dropout,
                                           emb_sizes=(512, 320, 128, 64))
        self.mca = MCAttention(heads=heads, dropout=dropout, emb_sizes=(256, 128, 320, 512))

    def forward(self, x00_detail, x00_context, x10_detail, x10_context, x20_detail, x20_context, x30_detail,
                x30_context) -> torch.Tensor:
        x00_detail, x10_detail, x20_detail, x30_detail = self.mca(x00_detail=x00_detail, x00_context=x00_context,
                                                                  x10_detail=x10_detail, x10_context=x10_context,
                                                                  x20_detail=x20_detail, x20_context=x20_context,
                                                                  x30_detail=x30_detail, x30_context=x30_context)
        return self.lv2decoder(x00_detail=x00_detail, x10_detail=x10_detail, x20_detail=x20_detail,
                               x30_detail=x30_detail)


class DualDecoderSwin(torch.nn.Module):
    def __init__(self, heads, feature, out_channels, dropout):
        super().__init__()
        self.lv2decoder = UnetDecoderCross(feature, out_channels, dropout, emb_sizes=(768, 384, 192, 96))
        self.mca = MCAttention(heads=heads, dropout=dropout, emb_sizes=(384, 192, 384, 768))

    def forward(self, x00_detail, x00_context, x10_detail, x10_context, x20_detail, x20_context, x30_detail,
                x30_context) -> torch.Tensor:
        x00_detail, x10_detail, x20_detail, x30_detail = self.mca(x00_detail=x00_detail, x00_context=x00_context,
                                                                  x10_detail=x10_detail, x10_context=x10_context,
                                                                  x20_detail=x20_detail, x20_context=x20_context,
                                                                  x30_detail=x30_detail, x30_context=x30_context)
        return self.lv2decoder(x00_detail=x00_detail, x10_detail=x10_detail, x20_detail=x20_detail,
                               x30_detail=x30_detail)


class UnetDecoderSingleMiTB1(torch.nn.Module):
    def __init__(self, feature, out_channels, dropout) -> None:
        """
        for MiT emb_sizes = (512,320,128,64)
        """
        super().__init__()
        self.x31_upconv = nn.ConvTranspose2d(feature, feature, kernel_size=(2, 2), stride=(2, 2))
        self.x31_doubleconv = DoubleConv(512, feature, drop_out=dropout)
        self.x22_upconv = nn.ConvTranspose2d(feature, feature, kernel_size=(2, 2), stride=(2, 2))
        self.x22_doubleconv = DoubleConv(320 + feature, feature, drop_out=dropout)
        self.x13_upconv = nn.ConvTranspose2d(feature, feature, kernel_size=(2, 2), stride=(2, 2))
        self.x13_doubleconv = DoubleConv(128 + feature, feature, drop_out=dropout)
        self.x04_upconv = nn.ConvTranspose2d(feature, feature, kernel_size=(2, 2), stride=(2, 2))
        self.x04_doubleconv = DoubleConv(64 + feature, feature, drop_out=dropout)

        self.intermedfinal_upconv = nn.ConvTranspose2d(feature, feature, kernel_size=(2, 2), stride=(2, 2))
        self.intermedfinal_doubleconv = DoubleConv(feature, feature, drop_out=dropout)
        self.final_upconv = nn.ConvTranspose2d(feature, feature, kernel_size=(2, 2), stride=(2, 2))

        self.final_doubleconv = DoubleConv(feature, feature, drop_out=dropout)
        self.segmentation_head = nn.Conv2d(feature, out_channels, kernel_size=(1, 1))  # segmentation output layer

    def forward(self, x00, x10, x20, x30) -> torch.Tensor:
        x31 = self.x31_doubleconv(x30)
        x31 = self.x31_upconv(x31)  # bottlekneck hence a little different

        x22 = torch.cat((x20, x31), dim=1)
        x22 = self.x22_doubleconv(x22)

        x13 = self.x13_upconv(x22)
        x13 = torch.cat((x10, x13), dim=1)
        x13 = self.x13_doubleconv(x13)

        x04 = self.x04_upconv(x13)
        x04 = torch.cat((x00, x04), dim=1)
        x04 = self.x04_doubleconv(x04)

        x0interfinal = self.intermedfinal_upconv(x04)
        x0interfinal = self.intermedfinal_doubleconv(x0interfinal)

        x0final = self.final_upconv(x0interfinal)
        x0final = self.final_doubleconv(x0final)
        return self.segmentation_head(x0final)


class UnetDecoderSingleSwin(torch.nn.Module):
    def __init__(self, feature, out_channels, dropout) -> None:
        """
        The Swin emb_sizes = (768,384,192,96)
        """
        super().__init__()
        self.x31_upconv = nn.ConvTranspose2d(feature, feature, kernel_size=(2, 2), stride=(2, 2))
        self.x31_doubleconv = DoubleConv(768, feature, drop_out=dropout)
        self.x22_upconv = nn.ConvTranspose2d(feature, feature, kernel_size=(2, 2), stride=(2, 2))
        self.x22_doubleconv = DoubleConv(384 + feature, feature, drop_out=dropout)
        self.x13_upconv = nn.ConvTranspose2d(feature, feature, kernel_size=(2, 2), stride=(2, 2))
        self.x13_doubleconv = DoubleConv(192 + feature, feature, drop_out=dropout)
        self.x04_upconv = nn.ConvTranspose2d(feature, feature, kernel_size=(2, 2), stride=(2, 2))
        self.x04_doubleconv = DoubleConv(96 + feature, feature, drop_out=dropout)

        self.intermedfinal_upconv = nn.ConvTranspose2d(feature, feature, kernel_size=(2, 2), stride=(2, 2))
        self.intermedfinal_doubleconv = DoubleConv(feature, feature, drop_out=dropout)
        self.final_upconv = nn.ConvTranspose2d(feature, feature, kernel_size=(2, 2), stride=(2, 2))

        self.final_doubleconv = DoubleConv(feature, feature, drop_out=dropout)
        self.segmentation_head = nn.Conv2d(feature, out_channels, kernel_size=(1, 1))  # segmentation output layer

    def forward(self, x00, x10, x20, x30) -> torch.Tensor:
        x31 = self.x31_doubleconv(x30)
        x31 = self.x31_upconv(x31)  # bottlekneck hence a little different

        x22 = torch.cat((x20, x31), dim=1)
        x22 = self.x22_doubleconv(x22)

        x13 = self.x13_upconv(x22)
        x13 = torch.cat((x10, x13), dim=1)
        x13 = self.x13_doubleconv(x13)

        x04 = self.x04_upconv(x13)
        x04 = torch.cat((x00, x04), dim=1)
        x04 = self.x04_doubleconv(x04)
        # if end the model here then Segformer output style
        x0interfinal = self.intermedfinal_upconv(x04)
        x0interfinal = self.intermedfinal_doubleconv(x0interfinal)

        x0final = self.final_upconv(x0interfinal)
        x0final = self.final_doubleconv(x0final)
        return self.segmentation_head(x0final)


class TestWeights(torch.nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        self.first_cross = MultiHeadCrossAttention(emb_size=2, num_heads=1, dropout=0.15)

    def forward(self, x00_detail, x00_context) -> torch.Tensor:
        return self.first_cross(x_detail=x00_detail, x_context=x00_context)


def test_one_weight():
    context_img = torch.randn(3, 2, 2)
    detail_img = torch.randn(3, 2, 2)
    print(detail_img)
    print(torch.sum(detail_img))
    model_test = TestWeights()

    model_test.first_cross.keys.weight = nn.Parameter(torch.eye(model_test.first_cross.keys.weight.shape[0]))

    model_test.first_cross.queries.weight = nn.Parameter(torch.zeros_like(model_test.first_cross.queries.weight))

    model_test.first_cross.values.weight = nn.Parameter(torch.eye(model_test.first_cross.values.weight.shape[0]))

    model_test.first_cross.projection.weight = nn.Parameter(
        torch.eye(model_test.first_cross.projection.weight.shape[0]))

    # change the bias
    model_test.first_cross.keys.bias = nn.Parameter(torch.zeros_like(model_test.first_cross.keys.bias))
    model_test.first_cross.queries.bias = nn.Parameter(torch.zeros_like(model_test.first_cross.queries.bias))
    model_test.first_cross.values.bias = nn.Parameter(torch.zeros_like(model_test.first_cross.values.bias))
    model_test.first_cross.projection.bias = nn.Parameter(torch.zeros_like(model_test.first_cross.projection.bias))

    model_test.eval()
    pred = model_test(detail_img, context_img)
    print(pred)


if __name__ == "__main__":
    test_one_weight()
    print("this file is only meant to be imported")
