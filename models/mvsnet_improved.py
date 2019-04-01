import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *


class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        self.inplanes = 32

        self.conv0 = ConvBnReLU(3, 8, 3, 1, 1)
        self.conv1 = ConvBnReLU(8, 8, 3, 1, 1)

        self.conv2 = ConvBnReLU(8, 16, 5, 2, 2)
        self.conv3 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv4 = ConvBnReLU(16, 16, 3, 1, 1)

        self.conv5 = ConvBnReLU(16, 32, 5, 2, 2)
        self.conv6 = ConvBnReLU(32, 32, 3, 1, 1)
        self.feature = nn.Conv2d(32, 32, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(self.conv0(x))
        x = self.conv4(self.conv3(self.conv2(x)))
        x = self.feature(self.conv6(self.conv5(x)))
        return x


# class ResFeature(nn.Module):
#     def __init__(self):
#         super(ResFeature, self).__init__()
#         self.inplanes = 32
#
#         self.conv1 = nn.Sequential(ConvBnReLU(3, 8, 3, 1, 1),
#                                    ConvBnReLU(8, 8, 3, 1, 1))
#         self.conv2 = self.make_res_layers(BasicBlock, 8, 16, 3, 2, 1)
#         self.conv3 = self.make_res_layers(BasicBlock, 16, 32, 3, 2, 1)
#
#         self.conv4 = nn.Conv2d(32, 32, 3, 1, 1)
#
#     def make_res_layers(self, block, in_channels, out_channels, num_blocks, stride, pad=1):
#         downsample = None
#         if stride != 1 or in_channels != out_channels:
#             downsample = ConvBn(in_channels, out_channels, 1, 1)
#         layers = list()
#         layers.append(block(in_channels, out_channels, stride, downsample, pad))
#         for i in range(1, num_blocks):
#             layers.append(block(out_channels, out_channels, 1, None, pad))
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         return x


class CostRegNet(nn.Module):
    def __init__(self):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(32, 8)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2)
        self.conv2 = ConvBnReLU3D(16, 16)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2)
        self.conv4 = ConvBnReLU3D(32, 32)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2)
        self.conv6 = ConvBnReLU3D(64, 64)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU())

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU())

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU())

        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        conv6 = self.conv6(self.conv5(conv4))
        conv8 = conv4 + self.conv7(conv6)
        conv10 = conv2 + self.conv9(conv8)
        conv12 = conv0 + self.conv11(conv10)
        prob = self.prob(conv12)
        return prob


# class CostRegularization(nn.Module):
#     def __init__(self):
#         super(CostRegularization, self).__init__()
#         self.conv1 = nn.Sequential(
#             ConvBnReLU3D(32, 16),
#             ConvBnReLU3D(16, 16))
#         self.conv2 = nn.Sequential(
#             ConvBnReLU3D(16, 8),
#             ConvBnReLU3D(8, 8))
#         self.hourglass1 = Hourglass3d(8)
#         self.conv3 = nn.Sequential(
#             ConvBnReLU3D(8, 8),
#             ConvBnReLU3D(8, 8))
#         self.conv4 = nn.Conv3d(8, 1, 3, stride=1, padding=1)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.hourglass1(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         return x

class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        self.conv1 = ConvBnReLU(4, 32)
        self.conv2 = ConvBnReLU(32, 32)
        self.conv3 = ConvBnReLU(32, 32)
        self.res = ConvBnReLU(32, 1)

    def forward(self, img, depth_init):
        concat = F.cat((img, depth_init), dim=1)
        depth_residual = self.res(self.conv3(self.conv2(self.conv1(concat))))
        depth_refined = depth_init + depth_residual
        return depth_refined


class MVSNet(nn.Module):
    def __init__(self, refine=True):
        super(MVSNet, self).__init__()
        self.refine = refine

        self.feature = FeatureNet()
        self.cost_regularization = CostRegNet()
        if self.refine:
            self.refine_network = RefineNet()

    def forward(self, imgs, proj_matrices, depth_values):
        imgs = torch.unbind(imgs, 1)
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(imgs) == len(proj_matrices), "Different number of images and projection matrices"
        img_height, img_width = imgs[0].shape[2], imgs[0].shape[3]
        num_depth = depth_values.shape[1]
        num_views = len(imgs)

        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        features = [self.feature(img) for img in imgs]
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # step 2. differentiable homograph, build cost volume
        # 2.1 warpped features
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        warped_volumes = [homo_warping(src_fea, src_proj, ref_proj, depth_values) for src_fea, src_proj in
                          zip(src_features, src_projs)]
        volumes = [ref_volume] + warped_volumes
        # 2.2 aggregate multiple feature volumes by variance
        avg_volume = sum(volumes) / num_views
        variance_volume = sum([(fea - avg_volume) ** 2 for fea in volumes]) / num_views

        # step 3. cost volume regularization
        cost_reg = self.cost_regularization(variance_volume)
        # cost_reg = F.upsample(cost_reg, [num_depth * 4, img_height, img_width], mode='trilinear')
        cost_reg = cost_reg.squeeze(1)
        prob_volume = F.softmax(cost_reg, dim=1)
        depth = depth_regression(prob_volume, depth_values=depth_values)

        # step 4. depth map refinement
        if not self.refine:
            return {"depth": depth}
        else:
            refined_depth = self.refine_network(depth)
            return {"depth": depth, "refined_depth": refined_depth}


def mvsnet_loss(depth_est, depth_gt, mask):
    mask = mask > 0.5
    return F.smooth_l1_loss(depth_est[mask], depth_gt[mask], size_average=True)
