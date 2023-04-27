# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
from numpy.linalg.linalg import solve

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


def  disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2) 
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        # self.nonlin = nn.ELU(inplace=True)
        self.nonlin = nn.ELU()

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """

    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """

    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)
        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)

        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)
        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """

    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        # print('projection')
        # print('points', points.shape)
        P = torch.matmul(K, T)[:, :3, :]
        # print('P', P.shape)

        cam_points = torch.matmul(P, points)
        # print('cam_points', cam_points.shape)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


class BackprojectDepthFishEye(nn.Module):
    """Layer to transform a depth image into a point cloud
    参考 https://github.com/valeoai/WoodScape/blob/5657a0c847e02a76c214eb84c61e618c1896ecf1/scripts/calibration/projection.py#L180
    """
    def __init__(self, batch_size, height, width, scale, gen_theta=False, eps=1e-7):
        super(BackprojectDepthFishEye, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

        self.scale = scale
        self.gen_theta = gen_theta
        self.distortion = np.array([1, 0, 0.0911736, 0, -0.000978799])
        theta_path =  "fisheye79_theta_5_scale_"+str(self.scale)+".npy"
        self.theta_path = theta_path
        if gen_theta == False and scale == 0:    
            print("load theta", theta_path, " scale", scale)
            self.theta = np.load(theta_path)
            self.theta_ori = torch.from_numpy(self.theta).cuda()
            self.eps = torch.tensor(self.eps).cuda()

    def prepare_cam_points(self, K):
        # for valeo dataset, aspect ratio is fx, fy, but in avp dataset, [fx, fy]        

        self.aspect_ratio = torch.tensor([K[0, 0, 2], K[0, 1,2]]).unsqueeze(-1).cuda()
        self.principle_point = torch.tensor([K[0,0,0], K[0,1,1]]).unsqueeze(-1).cuda()
        
        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        # id_coords shape: (2,h,w)
    
        self.pix_coords = np.stack([self.id_coords[0].view().reshape((-1)),
                                    self.id_coords[1].view().reshape((-1))], axis=0)
        # self.pix_coords = np.repeat(self.pix_coords[np.newaxis, :, :], self.batch_size, axis=0)
        self.pix_coords = torch.from_numpy(self.pix_coords).cuda()
        # 计算基于主点与比例的偏移
        self.cam_points_ori = (self.pix_coords - self.principle_point) \
                        / self.aspect_ratio   #(2, h*w) -(2,1)  # 这里相减时候会自动扩展



    def forward(self, depth, K):
        self.prepare_cam_points(K)
        
        self.cam_points_expand = self.cam_points_ori.unsqueeze(0)
        # print("cam_expand ",self.cam_points_expand.shape)
        self.cam_points = self.cam_points_expand.repeat(depth.shape[0], 1, 1)  #[batch_Size, 2, h*w]

        # 计算 r(theta)
        self.r_theta = (torch.norm(self.cam_points, dim=1)).unsqueeze(1)
        # theta,r_theta shape: (batch_size,1,h*w)
        # cam_points shape: (batch_size,2,h*w)
        # 通过深度估计的结果将鱼眼像素投影到3d坐标系
        self.theta_expand = self.theta_ori.unsqueeze(0).unsqueeze(1)
        self.theta = self.theta_expand.repeat(depth.shape[0], 1, 1)

        # print("x shape", x.shape, "self.theta", self.theta.shape) #[6, 1, 409600]) self.theta torch.Size([6, 1, 409600]
        rc = (depth.view((depth.shape[0], 1, -1)) * torch.sin(self.theta)).cuda()
        zc = (depth.view((depth.shape[0], 1, -1)) * torch.cos(self.theta)).cuda()
        # depth,rc,zc shape: (batch_size,1,h*w)
        xy = rc / (self.r_theta + self.eps) * self.cam_points
        # xy shape: (batch_size,2,h*w)
        self.ones = torch.ones(depth.shape[0], 1, self.height * self.width).cuda()
        world_points = (torch.cat([xy, zc, self.ones], 1)).float()
        # world_points shape: (batch_size,4,h*w)
        return world_points


class Project3DFishEye(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """

    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3DFishEye, self).__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps
        self.distortion = torch.tensor([1, 0, 0.0911736, 0, -0.000978799])

    def forward(self, points, K, T):
        self.aspect_ratio = torch.tensor([K[0, 0, 2], K[0, 1,2]]).cuda()
        self.principle_point = torch.tensor([K[0,0,0], K[0,1,1]]).cuda()
        # 计算位姿偏移
        P = T[:, :3, :]
        # P shape (batch_size, 3, 4)
        # points shape (batch_size, 4, (w*h))
        cam_points = torch.matmul(P, points)
        # cam_points shape (batch_size, 3, w*h)
        # 投影到鱼眼图像坐标系
        # 计算rc
        rc = torch.sqrt(torch.pow(cam_points[:, 0, :], 2) + torch.pow(cam_points[:, 1, :], 2))
        # 计算theta和r(theta)
        theta = ((np.pi / 2)) - torch.atan2(cam_points[:, 2, :], rc)
        r_theta = self.distortion[0] * theta + self.distortion[1] * torch.pow(theta, 2) + \
                  self.distortion[2] * torch.pow(theta, 3) + self.distortion[3] * torch.pow(theta, 4)
        # rc,theta,r_theta shape (batch_size, 1, w*h)
        # print(theta, r_theta)
        # 计算鱼眼坐标
        pix_coords = cam_points[:, :2, :] * (r_theta / (rc + self.eps)).unsqueeze(1)
        # 计算基于主点与比例的偏移
        pix_coords = (pix_coords * (self.aspect_ratio.unsqueeze(0)).unsqueeze(2)) \
                     + (self.principle_point.unsqueeze(0)).unsqueeze(2)
        try:
            pix_coords = pix_coords.view(cam_points.shape[0], 2, self.height, self.width)
        except RuntimeError:
            print("cam points shape", cam_points.shape)
        # pix_coords shape (batch_size, 2, w, h)
        # reshape
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        # print(pix_coords)
        # pix_coords shape (batch_size, w, h, 2)
        # 归一化到-1,1以方便后续计算
        
        # origin_pixcoords = pix_coords.detach()
        # print("origin pixel coords", origin_pixcoords.min(), origin_pixcoords.max())

        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = ((pix_coords - 0.5) * 2).float()
        return pix_coords

def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")

def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        #torch.clamp(input, min, max)将input的值限制在[min, max]之间，并返回结果
        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
