import numpy as np
from layers import *
import PIL.Image as pil
from PIL import Image 
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch
import sys
import cv2

def k_change_from_crop(origin_h, origin_w, dist_h, dist_w):
    """
    origin_h = 1080
    origin_w = 1920
    dist_w = 1920
    dist_h = 1056
    """
    K = np.array([[431.46,0,954.101,0],
                            [  0.,431.732 ,541.389, 0],
                            [  0  ,0, 1, 0],
                            [  0  ,0, 0, 1]])
    K[0,1] -= (origin_w - dist_w)/2         
    K[0,2] -= (origin_h - dist_h)/2

    print("K", K)
    return K

def k_change_from_scale(K, scale):
    K[0, :] = K[0, :]/scale
    K[1, :] = K[1, :]/scale
    return K


def test_image_projct_unproject(scale, gen_theta=False):
    print("scale is ", scale)
    origin_h = 1080
    origin_w = 1920
    dist_w = 1600
    dist_h = 1024


    h = dist_h // (2 ** scale)
    w = dist_w // (2 ** scale)
    # 对于不同的scale, K也不同，dataloader中的K是长宽归一化之后的
    # K = {
    #     'principle_point': np.array([954.101, 541.389], dtype=float)/(2 ** scale),
    #     'aspect_ratio': np.array([431.46, 431.732], dtype=float)/(2 ** scale),
    #     'distortion': np.array([1, 0, 0.0911736, 0, -0.000978799])
    #     # [1, 0, 0.0911736, 0, -0.000978799, 0, 0.00343315, 0, -0.00192034] for 1, 3, 5, 7 ,9
    #     }
    K = k_change_from_crop(origin_h, origin_w, dist_h, dist_w)
    # normalize
    K = k_change_from_scale(K, 2**scale)

    K = np.expand_dims(K, axis=0)

    # inv_K = np.linalg.pinv(K)
    # inputs[("K", scale)] = torch.from_numpy(K)
    # inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

    # K 只需要是tensor， depth 和T需要放到cuda上
    K = torch.from_numpy(K)

    up=BackprojectDepthFishEye(16, h,w, scale-1, gen_theta)
    down=Project3DFishEye(16,h,w)
    T=np.eye(4,dtype=np.float32)
    # print("numpy",T.dtype)
    T=np.repeat(T[np.newaxis, :, :], 16, axis=0)
    T = torch.from_numpy(T).cuda()

    # assume this is input_image's depth
    depth=torch.ones((16,h,w)).cuda()
    print(depth.shape)
    world_points=up(depth,K)
    # print(world_points,world_points.shape)
    print("world points", world_points.dtype, "T ", T.dtype)
    new_pix=down(world_points,K,T)
    new_pix = new_pix.cpu()
    # print(new_pix,new_pix.shape)

    image_path = "/home/data/avp_mini/fisheye79/1577851049000000000.jpg"
    input_image = pil.open(image_path).convert('RGB')
    # crop image
    input_image = input_image.crop(((origin_w-dist_w)/2,0,origin_w-(origin_w-dist_w)/2, dist_h))

    #TODO: down sample input image
    resize = transforms.Resize((h,w), interpolation=Image.ANTIALIAS)
    input_image = resize(input_image)
    input_image_pytorch = transforms.ToTensor()(input_image).unsqueeze(0)

    new_pix=new_pix.float()[0].unsqueeze(0)
    #torch.nn.functional.grid_sample(input, grid, mode='bilinear'
    #据grid中每个位置提供的坐标信息(这里指input中pixel的坐标)，将input中对应位置的像素值填充到grid指定的位置，得到最终的输出
    reconstruct_image = F.grid_sample(
                        input_image_pytorch,
                        new_pix,
                        padding_mode="border")


    # plt.imshow(input_image)
    input_image.save("input_image_scale"+str(scale)+".jpg")

    print(reconstruct_image[0].shape)
    new_img=transforms.ToPILImage()(reconstruct_image[0])
    # plt.imshow(new_img)
    new_img.save("new_image_scale"+str(scale)+".jpg")


if __name__ == "__main__":
    scales = [0,1,2,3,4]
    test_image_projct_unproject(1, gen_theta=False)
