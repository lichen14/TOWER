from __future__ import print_function
import math
import os
# from SimpleITK.SimpleITK import MaskedFFTNormalizedCorrelation
import torch
import random
import copy
import scipy
import imageio
import string
import torchvision.transforms as transforms
import numpy as np
import SimpleITK as sitk
from skimage.transform import resize
try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb
from torchvision.utils import make_grid
from skimage.transform import resize
import cv2
import matplotlib.pyplot as plt # plt 用于显示图片
from raw_mask_generation import *
def bernstein_poly(i, n, t):
    """
    伯恩斯坦多项式的递归定义
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       Control points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = torch.Tensor([p[0] for p in points])
    yPoints = torch.Tensor([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = torch.Tensor([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])
    # print(xPoints,polynomial_array.shape)
    # xvals = np.dot(xPoints, polynomial_array)
    # yvals = np.dot(yPoints, polynomial_array)
    xvals = torch.matmul(xPoints, polynomial_array)
    yvals = torch.matmul(yPoints, polynomial_array)
    # print(xvals.shape,yvals.shape)
    return xvals, yvals

def classification_data_augmentation(x,prob=0.5):
    # augmentation by flipping
    cnt = 3
    # print(x.shape,y.shape)
    while random.random() < prob and cnt > 0:
        degree = random.choice([[3], [2]])
        x = torch.flip(x, dims=degree)
        cnt = cnt - 1

    return x

def augmentation_0813(x,prob=0.5):
    gt=copy.deepcopy(x)
    rotation_transform = transforms.RandomRotation(90)
    pad_transform = transforms.Pad([1,1,1,1])

    resize_transform = transforms.Resize([28,28])
    horizontal_transform = transforms.RandomHorizontalFlip(1)
    vertical_transform = transforms.RandomVerticalFlip(1)
    
    rand=random.random()
    if  rand< 0.25:
        #print(rand)
        x = rotation_transform(x)
    elif rand<0.5:
        #print(rand)
        x = pad_transform(x)
        x = resize_transform(x)
    elif rand<0.75:
        #print(rand)
        x = horizontal_transform(x)
    else:
        #print(rand)
        x = vertical_transform(x)

    return x,gt

def  data_augmentation(x, y, prob=0.5):
    # augmentation by flipping
    # print(x.shape,y.shape)
    cnt = 3
    # print(x.shape,y.shape)
    while random.random() < prob and cnt > 0:
        degree = random.choice([[-1], [-2]])
        x = torch.flip(x, dims=degree)
        y = torch.flip(y, dims=degree)
        cnt = cnt - 1

    return x, y

def nonlinear_transformation(x, prob=0.5):
    # x=x.numpy()
    # if random.random() >= prob:
    #     return x
    # points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
    points = random.choice([[[0, 0], [0,0], [1,1], [1, 1]],[[0, 0], [0,1], [1,0], [1, 1]],[[0, 0], [0.25,0.75], [0.75,0.25], [1, 1]]])#[[0, 0], [0.25,0.75], [0.75,0.25], [1, 1]]#[[0, 0], [0,1], [1,0], [1, 1]]#random.choice([[[0, 0], [0.25,0.75], [0.75,0.25], [1, 1]], [[0, 0], [0,1], [1,0], [1, 1]]])#[[0, 0], [0.25,0.75], [0.75,0.25], [1, 1]]#[[0, 0], [0,1], [1,0], [1, 1]]#[[0, 1], [0,0], [1,1], [1, 0]]#[[0, 1], [0.25,0.25], [0.75,0.75], [1, 0]]#[[0, 0], [0,1], [1,0], [1, 1]]#[[0, 0], [0.25,0.75], [0.75,0.25], [1, 1]]#[[0, 0], [0,1], [1,0], [1, 1]]#[[0, 0], [0,1], [1,0], [1, 1]]#[[0, 0], [0.25,0.75], [0.75,0.25], [1, 1]]#[[0, 0], [0,0], [1,1], [1, 1]]#[[0, 0], [0.25,0.75], [0.75,0.25], [1, 1]]#
    xpoints = [p[0] for p in points]
    ypoints = [p[1] for p in points]
    xvals, yvals = bezier_curve(points, nTimes=1000)
    # print('points',points)
    if random.random() < 0.5:
        # Half change to get flip
        xvals,xindex = torch.sort(xvals)
    else:
        xvals,xindex = torch.sort(xvals)
        yvals,yindex = torch.sort(yvals)
    # print('xvals,yvals',xvals,yvals)  #(3, 512, 512)
    nonlinear_x = np.interp(x, xvals, yvals)
    # print(nonlinear_x.size, points, xvals.shape, yvals.shape)
    # x_ticks= np.arange(0,1.25,0.25)
    # plt.plot(xvals,yvals,linewidth=4)
    # plt.plot(xpoints,ypoints,'o',color ='red')
    # plt.grid() # 不显示坐标轴
    # plt.show()
    return nonlinear_x#, xvals, yvals]

def local_pixel_shuffling(x, prob=0.5):
    if random.random() >= prob:
        return x
    image_temp = copy.deepcopy(x)
    orig_image = copy.deepcopy(x)
    img_deps, img_rows, img_cols  = x.shape
    num_block = 50#5000
    for _ in range(num_block):
        block_noise_size_x = random.randint(1, 1)#img_rows//10)
        block_noise_size_y = random.randint(1, 1)#img_cols//10)
        # block_noise_size_z = random.randint(1, img_deps//10)
        block_noise_size_z = img_deps
        for c in range(img_deps):
            noise_x = random.randint(0, img_rows-block_noise_size_x)
            noise_y = random.randint(0, img_cols-block_noise_size_y)
            noise_z = img_deps#random.randint(0, img_deps-block_noise_size_z)
            window = orig_image[c,noise_x:noise_x+block_noise_size_x, 
                                noise_y:noise_y+block_noise_size_y]
            # print(c,window.shape)
            window = window.flatten()
            np.random.shuffle(window)
            # print(c,window.shape)
            window = window.reshape((block_noise_size_x, 
                                    block_noise_size_y
                                    ))
            image_temp[c,noise_x:noise_x+block_noise_size_x, 
                        noise_y:noise_y+block_noise_size_y] = window
    local_shuffling_x = image_temp

    return local_shuffling_x

def image_in_painting(x):
    img_deps,img_rows, img_cols = x.shape
    cnt = 5
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = random.randint(img_rows//6, img_rows//3)
        block_noise_size_y = random.randint(img_cols//6, img_cols//3)
        block_noise_size_z = img_deps#random.randint(img_deps//6, img_deps//3)
        noise_x = random.randint(3, img_rows-block_noise_size_x-3)
        noise_y = random.randint(3, img_cols-block_noise_size_y-3)
        noise_z = img_deps#random.randint(3, img_deps-block_noise_size_z-3)
        x[:,noise_x:noise_x+block_noise_size_x, 
          noise_y:noise_y+block_noise_size_y] = np.random.rand(img_deps,block_noise_size_x, 
                                                               block_noise_size_y) * 1.0
        cnt -= 1
    return x

def image_out_painting(x):
    img_deps,img_rows, img_cols = x.shape
    image_temp = copy.deepcopy(x)
    x = np.random.rand(x.shape[0], x.shape[1], x.shape[2]) * 1.0
    block_noise_size_x = img_rows - random.randint(3*img_rows//7, 4*img_rows//7)
    block_noise_size_y = img_cols - random.randint(3*img_cols//7, 4*img_cols//7)
    block_noise_size_z = img_deps#img_deps - random.randint(3*img_deps//7, 4*img_deps//7)
    noise_x = random.randint(3, img_rows-block_noise_size_x-3)
    noise_y = random.randint(3, img_cols-block_noise_size_y-3)
    noise_z = img_deps#random.randint(3, img_deps-block_noise_size_z-3)
    x[:,noise_x:noise_x+block_noise_size_x, 
      noise_y:noise_y+block_noise_size_y] = image_temp[:,noise_x:noise_x+block_noise_size_x, 
                                                        noise_y:noise_y+block_noise_size_y]
    cnt = 4
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = img_rows - random.randint(3*img_rows//7, 4*img_rows//7)
        block_noise_size_y = img_cols - random.randint(3*img_cols//7, 4*img_cols//7)
        block_noise_size_z = img_deps#img_deps - random.randint(3*img_deps//7, 4*img_deps//7)
        noise_x = random.randint(3, img_rows-block_noise_size_x-3)
        noise_y = random.randint(3, img_cols-block_noise_size_y-3)
        # noise_z = img_deps#random.randint(3, img_deps-block_noise_size_z-3)
        x[:,noise_x:noise_x+block_noise_size_x, 
            noise_y:noise_y+block_noise_size_y] = image_temp[:,noise_x:noise_x+block_noise_size_x, 
                                                                noise_y:noise_y+block_noise_size_y]
        cnt -= 1
    return x

def twist(image, numcontrolpoints, stdDef):
    sitkImage=sitk.GetImageFromArray(image, isVector=False)

    transfromDomainMeshSize=[numcontrolpoints]*sitkImage.GetDimension()

    tx = sitk.BSplineTransformInitializer(sitkImage,transfromDomainMeshSize)

    params = tx.GetParameters()

    paramsNp=np.asarray(params,dtype=float)
    paramsNp = paramsNp + np.random.randn(paramsNp.shape[0])*stdDef

    p = stdDef * np.random.randn(numcontrolpoints+3, numcontrolpoints+3, 2)
    p[:, 0, :] = 0
    p[:, -1, :] = 0
    p[0, :, :] = 0
    p[-1, :, :] = 0
    params=tuple(paramsNp)
    tx.SetParameters(p.flatten())

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitkImage)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(tx)

    resampler.SetDefaultPixelValue(0)
    outimgsitk = resampler.Execute(sitkImage)

    outimg = sitk.GetArrayFromImage(outimgsitk)

    return outimg

def twist_transformation(x, prob=0.5):
    twist_min, twist_max = 10, 20
    if random.random() >= prob:
        orig_image = copy.deepcopy(x)
        params1, params2 = int(random.random()*(twist_max-twist_min))+twist_min, int(random.random()*(twist_max-twist_min))+twist_min
        # print('twist!')
        x[0,:,:] = twist(orig_image[0,:,:] , params1, params2)
        x[1,:,:] = twist(orig_image[1,:,:] , params1, params2)
        x[2,:,:] = twist(orig_image[2,:,:] , params1, params2)
    
    return x

def StripeMaskingGenerator(x, prob=0.4):
    # if random.random() <= 0.3:
    #     return x,None
    orig_image = copy.deepcopy(x)
    img_deps,img_rows, img_cols = orig_image.shape  #(3, 512, 512)
    num_patches = img_rows* img_cols 
    num_mask = int(prob * num_patches)
    # print("mask generating: total patches {}, mask patches {}".format(num_patches,num_mask))
    mask = np.hstack(
        [np.ones(num_patches-num_mask),
         np.zeros(num_mask),]
    ).reshape(img_rows,img_cols)#.astype(bool)
    # print('mask',mask.shape)
    # mask = np.array(mask, dtype=bool)
    # print(mask)
    np.random.shuffle(mask)
    # print(mask.shape)
    for c in range(img_deps):
        # print(mask.reshape(img_rows,img_cols))
        new_img = orig_image[c]
        # print(new_img.shape)  #(262144,) (3, 512, 512)
        x[c] = new_img*mask
    # print('new_img',new_img)
    # print('x[2]',x[2])
    # out = orig_image*mask
    # x = np.dot(mask, orig_image)
    return x,mask

def BlockMaskingGenerator(x, prob=0.5):
    # if random.random() <= 0.2:
    #     return x,None
    orig_image = copy.deepcopy(x)
    img_deps,img_rows, img_cols = orig_image.shape  #(3, 512, 512)
    num_patches = img_rows* img_cols 
    num_mask = int(prob * num_patches)
    mask_rows = 32
    mask_cols = 32
    num_patches = mask_rows* mask_cols 
    num_mask = int(prob * num_patches)
    # print("mask generating: total patches {}, mask patches {}".format(num_patches,num_mask))
    mask = np.zeros([mask_rows, mask_cols])#np.random.rand(mask_rows, mask_cols)
    # print(mask.shape)
    # np.hstack(
    #     [np.ones(num_patches-num_mask),
    #      np.zeros(num_mask),]
    # ).reshape(mask_rows,mask_cols)#.astype(bool)
    # print('mask',mask)
    # np.random.shuffle(mask)
    mask=np.reshape(mask,(num_patches,1))
    # print(mask.shape)
    mask[:num_mask] =1
    mask[num_mask:] = 0
    np.random.shuffle(mask)
    mask=np.reshape(mask,(mask_rows, mask_cols))
    # print('mask',mask)
    
    # print(mask.shape)
    mask = cv2.resize(mask, (img_rows, img_cols),interpolation=cv2.INTER_NEAREST)
    # mask = cv2.resize(mask, dsize=None,fx=16,fy=16,interpolation=cv2.INTER_LINEAR)
    # mask = np.resize(mask, (img_rows, img_cols))
    # print(mask.shape)
    mask[mask>0.5] =1
    mask[mask<0.5]=0
    # mask[mask==2]=1
    # mask[mask>0] = 1
    for c in range(img_deps):
        # print(mask.reshape(img_rows,img_cols))
        new_img = orig_image[c]
        # print(new_img.shape)  #(262144,) (3, 512, 512)
        x[c] = new_img*mask
    # print('new_img',new_img)
    # print('x[2]',x[2])
    # out = orig_image*mask
    # x = np.dot(mask, orig_image)
    return x,mask

def generate_pair_0222(img, batch_size, config, status="test"):
    # print(img.shape)    #(Batch_Size, channels, rpws, cols)
    img_deps =1
    img_rows, img_cols = img.shape[1], img.shape[2]
    # img_deps, img_rows, img_cols = img.shape[1], img.shape[2], img.shape[3]
    # while True:
    # index = [i for i in range(img.shape[0])]
    # random.shuffle(index)
    # y = img[index[:batch_size]]
    block_mask = []
    new_img = copy.deepcopy(img)
    for n in range(img.shape[0]):
        
        # Autoencoder
        new_img[n] = copy.deepcopy(img[n])

        

        # Local Shuffle Pixel
        # new_img[n] = local_pixel_shuffling(new_img[n], prob=config.local_rate)
        
        # Apply non-Linear transformation with an assigned probability
        # new_img[n] = nonlinear_transformation(new_img[n], config.nonlinear_rate)
        
        # twist
        # new_img[n] = twist_transformation(new_img[n], config.twist_rate)

        # random masking-stipe
        # new_img[n] =  StripeMaskingGenerator(new_img[n], config.mask_ratio)

        # random masking-block
        new_img[n], block =  BlockMaskingGenerator(new_img[n], config.mask_ratio)
        block_mask.append(block)
        # Inpainting & Outpainting
        # if random.random() < config.paint_rate:
        #     if random.random() < config.inpaint_rate:
        #         # Inpainting
        #         new_img[n] = image_in_painting(new_img[n])
        #     else:
        #         # Outpainting
        #         new_img[n] = image_out_painting(new_img[n])
    # print(new_img.shape) 
    # Save sample images module
    if config.save_samples is not None and status == "train" and random.random() < 0.01:
        n_sample = random.choice( [i for i in range(config.batch_size)] )
        sample_1 = np.concatenate((new_img[n_sample,0,:,:,2*img_deps//6], img[n_sample,0,:,:,2*img_deps//6]), axis=1)
        sample_2 = np.concatenate((new_img[n_sample,0,:,:,3*img_deps//6], img[n_sample,0,:,:,3*img_deps//6]), axis=1)
        sample_3 = np.concatenate((new_img[n_sample,0,:,:,4*img_deps//6], img[n_sample,0,:,:,4*img_deps//6]), axis=1)
        sample_4 = np.concatenate((new_img[n_sample,0,:,:,5*img_deps//6], img[n_sample,0,:,:,5*img_deps//6]), axis=1)
        final_sample = np.concatenate((sample_1, sample_2, sample_3, sample_4), axis=0)
        final_sample = final_sample * 255.0
        final_sample = final_sample.astype(np.uint8)
        file_name = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(10)])+'.'+config.save_samples
        imageio.imwrite(os.path.join(config.sample_path, config.exp_name, file_name), final_sample)

    return (new_img, img,block_mask)

def generate_pair_0221(img, batch_size, config, status="test"):
    # print(img.shape)    #(Batch_Size, channels, rpws, cols)
    img_deps =1
    img_rows, img_cols = img.shape[1], img.shape[2]
    # img_deps, img_rows, img_cols = img.shape[1], img.shape[2], img.shape[3]
    # while True:
    # index = [i for i in range(img.shape[0])]
    # random.shuffle(index)
    # y = img[index[:batch_size]]
    block_mask = []
    point_record = []
    xvals_record = []
    yvals_record = []
    new_img = copy.deepcopy(img)
    for n in range(img.shape[0]):
        
        # Autoencoder
        new_img[n] = copy.deepcopy(img[n])

        

        # Local Shuffle Pixel
        # new_img[n] = local_pixel_shuffling(new_img[n], prob=config.local_rate)
        
        # Apply non-Linear transformation with an assigned probability
        new_img[n],xvals,yvals = nonlinear_transformation(new_img[n], config.nonlinear_rate)
        # point_record.append(points)
        xvals_record.append(xvals)
        yvals_record.append(yvals)
        # twist
        # new_img[n] = twist_transformation(new_img[n], config.twist_rate)

        # random masking-stipe
        # new_img[n] =  StripeMaskingGenerator(new_img[n], config.mask_ratio)

        # random masking-block
        # new_img[n], block =  BlockMaskingGenerator(new_img[n], config.mask_ratio)
        # block_mask.append(block)
        # Inpainting & Outpainting
        # if random.random() < config.paint_rate:
        #     if random.random() < config.inpaint_rate:
        #         # Inpainting
        #         new_img[n] = image_in_painting(new_img[n])
        #     else:
        #         # Outpainting
        #         new_img[n] = image_out_painting(new_img[n])
    # print(new_img.shape) 
    # Save sample images module
    if config.save_samples is not None and status == "train" and random.random() < 0.01:
        n_sample = random.choice( [i for i in range(config.batch_size)] )
        sample_1 = np.concatenate((new_img[n_sample,0,:,:,2*img_deps//6], img[n_sample,0,:,:,2*img_deps//6]), axis=1)
        sample_2 = np.concatenate((new_img[n_sample,0,:,:,3*img_deps//6], img[n_sample,0,:,:,3*img_deps//6]), axis=1)
        sample_3 = np.concatenate((new_img[n_sample,0,:,:,4*img_deps//6], img[n_sample,0,:,:,4*img_deps//6]), axis=1)
        sample_4 = np.concatenate((new_img[n_sample,0,:,:,5*img_deps//6], img[n_sample,0,:,:,5*img_deps//6]), axis=1)
        final_sample = np.concatenate((sample_1, sample_2, sample_3, sample_4), axis=0)
        final_sample = final_sample * 255.0
        final_sample = final_sample.astype(np.uint8)
        file_name = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(10)])+'.'+config.save_samples
        imageio.imwrite(os.path.join(config.sample_path, config.exp_name, file_name), final_sample)

    return (new_img, img,point_record,xvals_record, yvals_record)#block_mask)

def generate_pair_new(img, batch_size, config, status="test"):
    # print(img.shape)    #(Batch_Size, channels, rpws, cols)
    img_deps =1
    img_rows, img_cols = img.shape[1], img.shape[2]
    # img_deps, img_rows, img_cols = img.shape[1], img.shape[2], img.shape[3]
    # while True:
    # index = [i for i in range(img.shape[0])]
    # random.shuffle(index)
    # y = img[index[:batch_size]]
    block_mask = []
    point_record = []
    new_img = copy.deepcopy(img)
    for n in range(img.shape[0]):
        
        # Autoencoder
        new_img[n] = copy.deepcopy(img[n])

        

        # Local Shuffle Pixel
        # new_img[n] = local_pixel_shuffling(new_img[n], prob=config.local_rate)
        
        
        # Apply non-Linear transformation with an assigned probability
        # new_img[n] = nonlinear_transformation(new_img[n], config.nonlinear_rate)

        # point_record.append(record)
        # twist
        # new_img[n] = twist_transformation(new_img[n], config.twist_rate)
        
        # Apply raw_mask_generation to funds images
        # new_img[n],mask = raw_mask_print(new_img[n], num_raws=config.num_raws,path=batch_size)
        # new_img[n],mask = raw_mask_generation(new_img[n], num_raws=config.num_raws)

        # random masking-stipe
        new_img[n], mask  =  StripeMaskingGenerator(new_img[n], config.mask_ratio)

        # random masking-block
        # new_img[n], mask =  BlockMaskingGenerator(new_img[n], config.mask_ratio)
        # print(block.shape)
        if mask is None:
            block_mask.append(np.zeros((img.shape[2], img.shape[3])))
        else:
            block_mask.append(mask)
        # Inpainting & Outpainting
        # if random.random() < config.paint_rate:
        #     if random.random() < config.inpaint_rate:
        #         # Inpainting
        #         new_img[n] = image_in_painting(new_img[n])
        #     else:
        #         # Outpainting
        #         new_img[n] = image_out_painting(new_img[n])
    # Save sample images module
    if config.save_samples is not None and status == "train" and random.random() < 0.01:
        n_sample = random.choice( [i for i in range(config.batch_size)] )
        sample_1 = np.concatenate((new_img[n_sample,0,:,:,2*img_deps//6], img[n_sample,0,:,:,2*img_deps//6]), axis=1)
        sample_2 = np.concatenate((new_img[n_sample,0,:,:,3*img_deps//6], img[n_sample,0,:,:,3*img_deps//6]), axis=1)
        sample_3 = np.concatenate((new_img[n_sample,0,:,:,4*img_deps//6], img[n_sample,0,:,:,4*img_deps//6]), axis=1)
        sample_4 = np.concatenate((new_img[n_sample,0,:,:,5*img_deps//6], img[n_sample,0,:,:,5*img_deps//6]), axis=1)
        final_sample = np.concatenate((sample_1, sample_2, sample_3, sample_4), axis=0)
        final_sample = final_sample * 255.0
        final_sample = final_sample.astype(np.uint8)
        file_name = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(10)])+'.'+config.save_samples
        imageio.imwrite(os.path.join(config.sample_path, config.exp_name, file_name), final_sample)

    return (new_img, img,block_mask)

def generate_pair(img, batch_size, config, status="test"):
    # print(img.shape)    #(Batch_Size, 1, rpws, cols, channels)
    img_rows, img_cols, img_deps = img.shape[1], img.shape[2], img.shape[3]
    while True:
        index = [i for i in range(img.shape[0])]
        random.shuffle(index)
        y = img[index[:batch_size]]
        x = copy.deepcopy(y)
        for n in range(batch_size):
            
            # Autoencoder
            x[n] = copy.deepcopy(y[n])
            
            # # Flip
            # x[n], y[n] = data_augmentation(x[n], y[n], config.flip_rate)

            # # Local Shuffle Pixel
            x[n] = local_pixel_shuffling(x[n], prob=config.local_rate)
            
            # Apply non-Linear transformation with an assigned probability
            x[n] = nonlinear_transformation(x[n], config.nonlinear_rate)
            
            # # Inpainting & Outpainting
            # if random.random() < config.paint_rate:
            #     if random.random() < config.inpaint_rate:
            #         # Inpainting
            #         x[n] = image_in_painting(x[n])
            #     else:
            #         # Outpainting
            #         x[n] = image_out_painting(x[n])

        # Save sample images module
        # if config.save_samples is not None and status == "train" and random.random() < 0.01:
        #     n_sample = random.choice( [i for i in range(config.batch_size)] )
        #     sample_1 = np.concatenate((x[n_sample,0,:,:,2*img_deps//6], y[n_sample,0,:,:,2*img_deps//6]), axis=1)
        #     sample_2 = np.concatenate((x[n_sample,0,:,:,3*img_deps//6], y[n_sample,0,:,:,3*img_deps//6]), axis=1)
        #     sample_3 = np.concatenate((x[n_sample,0,:,:,4*img_deps//6], y[n_sample,0,:,:,4*img_deps//6]), axis=1)
        #     sample_4 = np.concatenate((x[n_sample,0,:,:,5*img_deps//6], y[n_sample,0,:,:,5*img_deps//6]), axis=1)
        #     final_sample = np.concatenate((sample_1, sample_2, sample_3, sample_4), axis=0)
        #     final_sample = final_sample * 255.0
        #     final_sample = final_sample.astype(np.uint8)
        #     file_name = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(10)])+'.'+config.save_samples
        #     imageio.imwrite(os.path.join(config.sample_path, config.exp_name, file_name), final_sample)

        yield (x, y)
def draw_in_tensorboard_1channel(writer, gt, i_iter, images, pred, type_):
    grid_image = make_grid(gt.clone(), 3, normalize=True)
    writer.add_image(f'original Image - {type_}', grid_image, i_iter)

    # print('target_labels:',target_labels.shape)#[1, 256, 256]
    grid_image = make_grid(images.clone(), 3, normalize=True)
    writer.add_image(f'transformed Image - {type_}', grid_image, i_iter)

    grid_image = make_grid(pred.clone(), 3, normalize=True)
    writer.add_image(f'recovered Image - {type_}', grid_image, i_iter)

def draw_in_tensorboard(writer, gt, i_iter, images, pred, type_):
    grid_image = make_grid(gt[:3].clone(), 3, normalize=True)
    writer.add_image(f'original Image - {type_}', grid_image, i_iter)

    # print('target_labels:',target_labels.shape)#[1, 256, 256]
    grid_image = make_grid(images[:3].clone(), 3, normalize=True)
    writer.add_image(f'transformed Image - {type_}', grid_image, i_iter)

    grid_image = make_grid(pred[:3].clone(), 3, normalize=True)
    writer.add_image(f'recovered Image - {type_}', grid_image, i_iter)

def draw_in_tensorboard_vfs(name, gt, log_pth, images, pred, type_):
    if not os.path.exists(os.path.join(log_pth,'output-valid')):
        os.makedirs(os.path.join(log_pth,'output-valid'))     
    if not os.path.exists(os.path.join(log_pth,'transformation-valid')):
        os.makedirs(os.path.join(log_pth,'transformation-valid'))
    #print(name[0])
    # plt.subplot(221)
    # plt.imshow(np.transpose(gt.cpu().data,(1,2,0)))
    # plt.axis('off') # 不显示坐标轴
    # #plt.savefig(os.path.join(log_pth,'output',file_name),dpi=300.0,pad_inches=0.0)
    # plt.subplot(222)
    # plt.imshow(np.transpose(images.cpu().data,(1,2,0)))
    # plt.axis('off') # 不显示坐标轴
    # #plt.savefig(os.path.join(log_pth,'output',file_name),dpi=300.0,pad_inches=0.0)
    # plt.subplot(223)
    # plt.imshow(np.transpose(pred.cpu().data,(1,2,0)))
    # plt.show()
    pred_new = torch.cat((pred,pred,pred),dim=0)
    #print(pred_new.shape)
    #print(pred_new.shape)
    plt.imsave(os.path.join(log_pth,'output-valid',str(name[0]).replace("tif","jpg")),np.transpose(pred.cpu().data.numpy(),(1,2,0)))
    # plt.axis('off') # 不显示坐标轴
    # plt.savefig(os.path.join(log_pth,'output',file_name),dpi=300.0,pad_inches=0.0)
    #print(images.cpu().data.numpy().shape)
    plt.imsave(os.path.join(log_pth,'transformation-valid',str(name[0]).replace("tif","jpg")),np.transpose(images.cpu().data.numpy(),(1,2,0)))

def draw_in_tensorboard_vfs_0305(name, gt, log_pth, images, pred, type_):
    if not os.path.exists(os.path.join(log_pth,'output-valid')):
        os.makedirs(os.path.join(log_pth,'output-valid'))     
    if not os.path.exists(os.path.join(log_pth,'transformation-valid')):
        os.makedirs(os.path.join(log_pth,'transformation-valid'))
    #print(name[0])
    # plt.subplot(221)
    # plt.imshow(np.transpose(gt.cpu().data,(1,2,0)))
    # plt.axis('off') # 不显示坐标轴
    # #plt.savefig(os.path.join(log_pth,'output',file_name),dpi=300.0,pad_inches=0.0)
    # plt.subplot(222)
    # plt.imshow(np.transpose(images.cpu().data,(1,2,0)))
    # plt.axis('off') # 不显示坐标轴
    # #plt.savefig(os.path.join(log_pth,'output',file_name),dpi=300.0,pad_inches=0.0)
    # plt.subplot(223)
    # plt.imshow(np.transpose(pred.cpu().data,(1,2,0)))
    # plt.show()
    
    pred_new = torch.cat((pred,pred,pred),dim=0)
    #print(pred_new.shape)
    #print(pred_new.shape)
    for i in range(pred.shape[0]):

        plt.imsave(os.path.join(log_pth,'output-valid',str(name[i]).replace("tif","jpg")),np.transpose(pred.cpu().data[i].numpy(),(1,2,0)))
        # plt.axis('off') # 不显示坐标轴
        # plt.savefig(os.path.join(log_pth,'output',file_name),dpi=300.0,pad_inches=0.0)
        #print(images.cpu().data.numpy().shape)
        plt.imsave(os.path.join(log_pth,'transformation-valid',str(name[i]).replace("tif","jpg")),np.transpose(images.cpu().data[i].numpy(),(1,2,0)))
def draw_in_tensorboard_vfs_0306(name, mask, log_pth, images, pred, type_):
    if not os.path.exists(os.path.join(log_pth,'output-valid-L1')):
        os.makedirs(os.path.join(log_pth,'output-valid-L1'))     
    if not os.path.exists(os.path.join(log_pth,'transformation-valid-L1')):
        os.makedirs(os.path.join(log_pth,'transformation-valid-L1'))
    if not os.path.exists(os.path.join(log_pth,'transformation-mask-L1')):
        os.makedirs(os.path.join(log_pth,'transformation-mask-L1'))
    #print(name[0])
    # plt.subplot(221)
    # plt.imshow(np.transpose(gt.cpu().data,(1,2,0)))
    # plt.axis('off') # 不显示坐标轴
    # #plt.savefig(os.path.join(log_pth,'output',file_name),dpi=300.0,pad_inches=0.0)
    # plt.subplot(222)
    # plt.imshow(np.transpose(images.cpu().data,(1,2,0)))
    # plt.axis('off') # 不显示坐标轴
    # #plt.savefig(os.path.join(log_pth,'output',file_name),dpi=300.0,pad_inches=0.0)
    # plt.subplot(223)
    # plt.imshow(np.transpose(pred.cpu().data,(1,2,0)))
    # plt.show()
    pred_new = torch.cat((pred,pred,pred),dim=0)
    #print(pred_new.shape)
    # pred = torch.transpose(pred,1,2)
    # pred = torch.transpose(pred,2,3)
    # images = torch.transpose(images,1,2)
    # images = torch.transpose(images,2,3)
    #print(pred_new.shape)
    mask = torch.Tensor(mask)
    for i in range(pred.shape[0]):
        # print(name[i])
        tmp = mask[i]#.squeeze(0)
        index = tmp<1
        # for i in range(images.shape[1]):
        #     for j in range(images.shape[2]):
        block=np.zeros((512,512,3))
        # block = torch.cat((tmp.unsqueeze(0),tmp.unsqueeze(0),tmp.unsqueeze(0)),dim=0)
        # print(index.shape,images.shape)
        # images[:,index]=torch.Tensor([1,0,0])
        block[index,0]=1#255
        block[index,1]=0
        block[index,2]=0
        images[i,0,index]=1#255
        images[i,1,index]=0
        images[i,2,index]=0
        plt.imsave(os.path.join(log_pth,'output-valid-L1',os.path.basename(str(name[i]))),np.transpose(pred.cpu().data[i].numpy(),(1,2,0)))
        # plt.axis('off') # 不显示坐标轴
        # plt.savefig(os.path.join(log_pth,'output',file_name),dpi=300.0,pad_inches=0.0)
        #print(images.cpu().data.numpy().shape)
        plt.imsave(os.path.join(log_pth,'transformation-mask-L1',os.path.basename(str(name[i]))),block)
        plt.imsave(os.path.join(log_pth,'transformation-valid-L1',os.path.basename(str(name[i]))),np.transpose(images.cpu().data[i].numpy(),(1,2,0)))

def draw_in_tensorboard_txc_0307(name, mask, log_pth, images, pred, type_):
    if not os.path.exists(os.path.join(log_pth,'output-valid-L11')):
        os.makedirs(os.path.join(log_pth,'output-valid-L11'))     
    if not os.path.exists(os.path.join(log_pth,'transformation-valid-L11')):
        os.makedirs(os.path.join(log_pth,'transformation-valid-L11'))
    if not os.path.exists(os.path.join(log_pth,'transformation-mask-L11')):
        os.makedirs(os.path.join(log_pth,'transformation-mask-L11'))
    #print(name[0])
    # plt.subplot(221)
    # plt.imshow(np.transpose(gt.cpu().data,(1,2,0)))
    # plt.axis('off') # 不显示坐标轴
    # #plt.savefig(os.path.join(log_pth,'output',file_name),dpi=300.0,pad_inches=0.0)
    # plt.subplot(222)
    # plt.imshow(np.transpose(images.cpu().data,(1,2,0)))
    # plt.axis('off') # 不显示坐标轴
    # #plt.savefig(os.path.join(log_pth,'output',file_name),dpi=300.0,pad_inches=0.0)
    # plt.subplot(223)
    # plt.imshow(np.transpose(pred.cpu().data,(1,2,0)))
    # plt.show()
    # print(pred.shape)
    pred_new = torch.cat((pred,pred,pred),dim=1)
    # print(pred_new.shape)
    # pred = torch.transpose(pred,1,2)
    # pred = torch.transpose(pred,2,3)
    # images = torch.transpose(images,1,2)
    # images = torch.transpose(images,2,3)
    # print(pred.shape)
    mask = torch.Tensor(mask)
    for i in range(pred.shape[0]):
        tmp = mask[i]#.squeeze(0)
        index = (tmp<1)
        # for i in range(images.shape[1]):
        #     for j in range(images.shape[2]):
        block=np.zeros((224,224,3))
        # block = torch.cat((tmp.unsqueeze(0),tmp.unsqueeze(0),tmp.unsqueeze(0)),dim=0)
        # print(index.shape,images.shape)
        # block[index,:]=torch.Tensor([1,0,0])
        block[index,0]=1#255
        block[index,1]=0
        block[index,2]=0

        # index = (gt<1)
    # for i in range(images.shape[1]):
    #     for j in range(images.shape[2]):

        images[i,0,index]=1#255
        images[i,1,index]=0
        images[i,2,index]=0
        # print(pred_new[i].shape)
        # images[:,index]=1#255
        # images[1,index]=0
        # images[2,index]=0
        plt.imsave(os.path.join(log_pth,'output-valid-L11',os.path.basename(str(name[i]))),np.transpose(pred.cpu().data[i].numpy(),(1,2,0)))
        # plt.axis('off') # 不显示坐标轴
        # plt.savefig(os.path.join(log_pth,'output',file_name),dpi=300.0,pad_inches=0.0)
        #print(images.cpu().data.numpy().shape)
        plt.imsave(os.path.join(log_pth,'transformation-mask-L11',os.path.basename(str(name[i]))),block)
        plt.imsave(os.path.join(log_pth,'transformation-valid-L11',os.path.basename(str(name[i]))),np.transpose(images.cpu().data[i].numpy(),(1,2,0)))
      

def draw_in_tensorboard_mask(name, gt, log_pth, images, pred, type_):
    #print(name[0])
    if not os.path.exists(os.path.join(log_pth,'output-valid')):
        os.makedirs(os.path.join(log_pth,'output-valid'))     
    if not os.path.exists(os.path.join(log_pth,'transformation-valid')):
        os.makedirs(os.path.join(log_pth,'transformation-valid'))
    #print(name[0])
    # plt.subplot(221)
    # plt.imshow(np.transpose(gt.cpu().data,(1,2,0)))
    # plt.axis('off') # 不显示坐标轴
    # #plt.savefig(os.path.join(log_pth,'output',file_name),dpi=300.0,pad_inches=0.0)
    # plt.subplot(222)
    # plt.imshow(np.transpose(images.cpu().data,(1,2,0)))
    # plt.axis('off') # 不显示坐标轴
    # #plt.savefig(os.path.join(log_pth,'output',file_name),dpi=300.0,pad_inches=0.0)
    # plt.subplot(223)
    # plt.imshow(np.transpose(pred.cpu().data,(1,2,0)))
    # plt.show()
    gt = torch.Tensor(gt)
    index = (gt<1)
    # for i in range(images.shape[1]):
    #     for j in range(images.shape[2]):

    images[0,index]=1#255
    images[1,index]=0
    images[2,index]=0
    # pred_new = torch.cat((pred,pred,pred),dim=0)
    #print(pred.shape)
    #print(pred_new.shape)
    plt.imsave(os.path.join(log_pth,'output-valid',os.path.basename(str(name[0]))),np.transpose(pred.cpu().data.numpy(),(1,2,0)))
    # plt.axis('off') # 不显示坐标轴
    # plt.savefig(os.path.join(log_pth,'output',file_name),dpi=300.0,pad_inches=0.0)
    #print(images.cpu().data.numpy().shape)
    plt.imsave(os.path.join(log_pth,'transformation-valid',os.path.basename(str(name[0]))),np.transpose(images.cpu().data.numpy(),(1,2,0)))

def draw_in_tensorboard_mask_vfs(name, gt, log_pth, images, pred, type_):
    # print(name[0],pred.shape,images.shape)
    if not os.path.exists(os.path.join(log_pth,'output-valid')):
        os.makedirs(os.path.join(log_pth,'output-valid'))     
    if not os.path.exists(os.path.join(log_pth,'transformation-valid')):
        os.makedirs(os.path.join(log_pth,'transformation-valid'))
    #print(name[0])
    # plt.subplot(221)
    # plt.imshow(np.transpose(gt.cpu().data,(1,2,0)))
    # plt.axis('off') # 不显示坐标轴
    # #plt.savefig(os.path.join(log_pth,'output',file_name),dpi=300.0,pad_inches=0.0)
    # plt.subplot(222)
    # plt.imshow(np.transpose(images.cpu().data,(1,2,0)))
    # plt.axis('off') # 不显示坐标轴
    # #plt.savefig(os.path.join(log_pth,'output',file_name),dpi=300.0,pad_inches=0.0)
    # plt.subplot(223)
    # plt.imshow(np.transpose(pred.cpu().data,(1,2,0)))
    # plt.show()
    index = (gt<1)
    # for i in range(images.shape[1]):
    #     for j in range(images.shape[2]):

    images[0,index]=1#255
    images[1,index]=0
    images[2,index]=0
    # pred_new = torch.cat((pred,pred,pred),dim=0)
    #print(pred.shape)
    #print(pred_new.shape)
    plt.imsave(os.path.join(log_pth,'output-valid',os.path.basename(str(name[0]).replace("tif","jpg"))),np.transpose(pred.cpu().data.numpy(),(1,2,0)))
    # plt.axis('off') # 不显示坐标轴
    # plt.savefig(os.path.join(log_pth,'output',file_name),dpi=300.0,pad_inches=0.0)
    #print(images.cpu().data.numpy().shape)
    plt.imsave(os.path.join(log_pth,'transformation-valid',os.path.basename(str(name[0]).replace("tif","jpg"))),np.transpose(images.cpu().data.numpy(),(1,2,0)))

def draw_in_tensorboard_mask_dxc(name, gt, log_pth, images, pred, type_):
    # print(name[0],pred.shape,images.shape)
    if not os.path.exists(os.path.join(log_pth,'output-valid')):
        os.makedirs(os.path.join(log_pth,'output-valid'))     
    if not os.path.exists(os.path.join(log_pth,'transformation-valid')):
        os.makedirs(os.path.join(log_pth,'transformation-valid'))
    #print(name[0])
    # plt.subplot(221)
    # plt.imshow(np.transpose(gt.cpu().data,(1,2,0)))
    # plt.axis('off') # 不显示坐标轴
    # #plt.savefig(os.path.join(log_pth,'output',file_name),dpi=300.0,pad_inches=0.0)
    # plt.subplot(222)
    # plt.imshow(np.transpose(images.cpu().data,(1,2,0)))
    # plt.axis('off') # 不显示坐标轴
    # #plt.savefig(os.path.join(log_pth,'output',file_name),dpi=300.0,pad_inches=0.0)
    # plt.subplot(223)
    # plt.imshow(np.transpose(pred.cpu().data,(1,2,0)))
    # plt.show()
    index = (gt<1)
    # for i in range(images.shape[1]):
    #     for j in range(images.shape[2]):

    images[0,index]=1#255
    images[1,index]=0
    images[2,index]=0
    # pred_new = torch.cat((pred,pred,pred),dim=0)
    #print(pred.shape)
    #print(pred_new.shape)
    plt.imsave(os.path.join(log_pth,'output-valid',str(name[0]).replace("/","-")),np.transpose(pred.cpu().data.numpy(),(1,2,0)))
    # plt.axis('off') # 不显示坐标轴
    # plt.savefig(os.path.join(log_pth,'output',file_name),dpi=300.0,pad_inches=0.0)
    #print(images.cpu().data.numpy().shape)
    plt.imsave(os.path.join(log_pth,'transformation-valid',str(name[0]).replace("/","-")),np.transpose(images.cpu().data.numpy(),(1,2,0)))

def draw_in_tensorboard_txc(name, gt, log_pth, images, pred, type_):
    #print(name[0])
    if not os.path.exists(os.path.join(log_pth,'output-valid')):
        os.makedirs(os.path.join(log_pth,'output-valid'))     
    if not os.path.exists(os.path.join(log_pth,'transformation-valid')):
        os.makedirs(os.path.join(log_pth,'transformation-valid'))
    
    if not os.path.exists(os.path.join(log_pth,'transformation-mask')):
        os.makedirs(os.path.join(log_pth,'transformation-mask'))
    #print(name[0])
    # plt.subplot(221)
    # plt.imshow(np.transpose(gt.cpu().data,(1,2,0)))
    # plt.axis('off') # 不显示坐标轴
    # #plt.savefig(os.path.join(log_pth,'output',file_name),dpi=300.0,pad_inches=0.0)
    # plt.subplot(222)
    # plt.imshow(np.transpose(images.cpu().data,(1,2,0)))
    # plt.axis('off') # 不显示坐标轴
    # #plt.savefig(os.path.join(log_pth,'output',file_name),dpi=300.0,pad_inches=0.0)
    # plt.subplot(223)
    # plt.imshow(np.transpose(pred.cpu().data,(1,2,0)))
    # plt.show()
    index = (gt<1)
    # for i in range(images.shape[1]):
    #     for j in range(images.shape[2]):
    block=np.zeros((224,224,3))
        # block = torch.cat((tmp.unsqueeze(0),tmp.unsqueeze(0),tmp.unsqueeze(0)),dim=0)
        # print(index.shape,block.shape)
        # block[index,:]=torch.Tensor([1,0,0])
    block[index,0]=1#255
    block[index,1]=0
    block[index,2]=0
    images[0,index]=1#255
    images[1,index]=0
    images[2,index]=0
    
    pred_new = torch.cat((pred,pred,pred),dim=0)
    #print(pred.shape)
    #print(pred_new.shape)
    plt.imsave(os.path.join(log_pth,'output-valid',os.path.basename(str(name[0]))),np.transpose(pred_new.cpu().data.numpy(),(1,2,0)))
    # plt.axis('off') # 不显示坐标轴
    # plt.savefig(os.path.join(log_pth,'output',file_name),dpi=300.0,pad_inches=0.0)
    #print(images.cpu().data.numpy().shape)
    plt.imsave(os.path.join(log_pth,'transformation-mask',os.path.basename(str(name[0]))),block)
    plt.imsave(os.path.join(log_pth,'transformation-valid',os.path.basename(str(name[0]))),np.transpose(images.cpu().data.numpy(),(1,2,0)))

def draw_in_tensorboard_my(name, gt, log_pth, images, pred, type_):
    if not name[0].endswith('lateral.jpg'):
        if not os.path.exists(os.path.join(log_pth,'output-valid')):
            os.makedirs(os.path.join(log_pth,'output-valid'))     
        if not os.path.exists(os.path.join(log_pth,'transformation-valid')):
            os.makedirs(os.path.join(log_pth,'transformation-valid'))
        #plt.subplot(221)
        # plt.imshow(np.transpose(gt.cpu().data,(1,2,0)))
        # plt.axis('off') # 不显示坐标轴
        # plt.savefig(os.path.join(log_pth,'output',file_name),dpi=300.0,pad_inches=0.0)
        # #plt.subplot(222)
        # plt.imshow(np.transpose(images.cpu().data,(1,2,0)))
        # plt.axis('off') # 不显示坐标轴
        # plt.savefig(os.path.join(log_pth,'output',file_name),dpi=300.0,pad_inches=0.0)
        # #plt.subplot(223)
        # plt.imshow(np.transpose(pred.cpu().data,(1,2,0)))
        pred_new = torch.cat((pred,pred,pred),dim=0)
        #print(pred_new.shape)
        plt.imsave(os.path.join(log_pth,'output-valid',str(name[0]).replace("/","-")),np.transpose(pred_new.cpu().data.numpy(),(1,2,0)))
        # plt.axis('off') # 不显示坐标轴
        # plt.savefig(os.path.join(log_pth,'output',file_name),dpi=300.0,pad_inches=0.0)
        #print(images.cpu().data.numpy().shape)
        plt.imsave(os.path.join(log_pth,'transformation-valid',str(name[0]).replace("/","-")),np.transpose(images.cpu().data.numpy(),(1,2,0)))
    #plt.savefig(os.path.join(dir_name,'/output1'+file_name),dpi=300.0,pad_inches=0.0)
    #plt.show()
    # writer.add_image(f'original Image - {type_}', grid_image, i_iter)

    # # print('target_labels:',target_labels.shape)#[1, 256, 256]
    # grid_image = make_grid(images[:3].clone(), 3, normalize=True)
    # writer.add_image(f'transformed Image - {type_}', grid_image, i_iter)

    # grid_image = make_grid(pred[:3].clone(), 3, normalize=True)
    # writer.add_image(f'recovered Image - {type_}', grid_image, i_iter)

def log_losses_tensorboard(writer, current_losses, i_iter, type_):
    for loss_value in current_losses:
        writer.add_scalar(f'{type_}-loss', to_numpy(loss_value), i_iter)

def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()

def cosine_anneal_schedule(t,epochs,learning_rate):
    T=epochs
    M=1
    alpha_zero = learning_rate

    cos_inner = np.pi * (t % (T // M))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= T // M
    cos_out = np.cos(cos_inner) + 1
    return float(alpha_zero / 2 * cos_out)