import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class MorphologicalBoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        'Implementation of forward for this baseline experiment.'
        eroded_target = 1 - F.max_pool2d(1 - targets, kernel_size=3, stride=1, padding=1)
        boundary_mask = targets - eroded_target
        boundary_mask = F.max_pool2d(boundary_mask, kernel_size=3, stride=1, padding=1)

        pixel_bce = self.bce(logits, targets)
        boundary_loss = (pixel_bce * boundary_mask).sum() / (boundary_mask.sum() + 1e-8)
        return boundary_loss


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class AFF(nn.Module):
    def __init__(self, channels, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels), nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels), nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, outI, feature_Level):
        xa = x + outI + feature_Level
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        wei = self.sigmoid(xl + xg)
        return 2 * x * wei + 2 * outI * (1 - wei)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, AFF_channel, bilinear=False):
        super().__init__()
        self.aff = AFF(AFF_channel)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels // 2, out_channels)

    def forward(self, x1, x2, x3):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = self.aff(x2, x1, x3)
        return self.conv(x)


class AHFUnet_Baseline(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=False, dropout_rate=0.0, w_focal=0.4, w_dice=0.4,
                 w_bound=0.2):
        super(AHFUnet_Baseline, self).__init__()
        self.w_focal, self.w_dice, self.w_bound = w_focal, w_dice, w_bound
        self.focal_loss_fn = smp.losses.FocalLoss(mode='binary', alpha=0.75, gamma=2.0)
        self.dice_loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=True)
        self.boundary_loss_fn = MorphologicalBoundaryLoss()

        self.spatial_avgpool = nn.AdaptiveAvgPool2d(1)

        
        self.conv1 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Dropout(dropout_rate))
        self.trans_conv1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True), nn.Dropout(dropout_rate))
        self.trans_conv2 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True), nn.Dropout(dropout_rate))
        self.trans_conv3 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
        self.conv4 = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True), nn.Dropout(dropout_rate))
        self.trans_conv4 = nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1, output_padding=1)

        
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.up1 = Up(1024, 512, 512, bilinear=bilinear)
        self.up2 = Up(512, 256, 256, bilinear=bilinear)
        self.up3 = Up(256, 128, 128, bilinear=bilinear)
        self.up4 = Up(128, 64, 64, bilinear=bilinear)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x, st_data=None, st_mask=None, current_drop_rate=None, targets=None, tissue_masks=None):
        x1 = self.inc(x)
        s1, c1 = self.spatial_avgpool(x1), torch.mean(x1, dim=1, keepdim=True)

        x2 = self.down1(x1)
        s2, c2 = self.spatial_avgpool(x2), torch.mean(x2, dim=1, keepdim=True)
        I_out_1 = self.conv1(s1 * c1 + self.trans_conv1(s2 * c2)) * x1

        x3 = self.down2(x2)
        s3, c3 = self.spatial_avgpool(x3), torch.mean(x3, dim=1, keepdim=True)
        I_out_2 = self.conv2(s2 * c2 + self.trans_conv2(s3 * c3)) * x2

        x4 = self.down3(x3)
        s4, c4 = self.spatial_avgpool(x4), torch.mean(x4, dim=1, keepdim=True)
        I_out_3 = self.conv3(s3 * c3 + self.trans_conv3(s4 * c4)) * x3

        x5 = self.down4(x4)
        s5, c5 = self.spatial_avgpool(x5), torch.mean(x5, dim=1, keepdim=True)
        I_out_4 = self.conv4(s4 * c4 + self.trans_conv4(s5 * c5)) * x4

        x_dec = self.up1(x5, I_out_4, x4)
        x_dec = self.up2(x_dec, I_out_3, x3)
        x_dec = self.up3(x_dec, I_out_2, x2)
        x_dec = self.up4(x_dec, I_out_1, x1)
        logits = self.outc(x_dec)

        if targets is not None and tissue_masks is not None:
            masked_logits = torch.where(tissue_masks > 0.5, logits, torch.tensor(-10.0, device=logits.device))
            active_targets = targets * tissue_masks
            loss_focal = self.focal_loss_fn(masked_logits, active_targets)
            loss_dice = self.dice_loss_fn(masked_logits, active_targets)
            loss_bound = self.boundary_loss_fn(masked_logits, active_targets)
            total_loss = (self.w_focal * loss_focal) + (self.w_dice * loss_dice) + (self.w_bound * loss_bound)
            return logits, total_loss, {'loss': total_loss.item(), 'focal': loss_focal.item(), 'dice': loss_dice.item(),
                                        'bound': loss_bound.item()}

        return logits