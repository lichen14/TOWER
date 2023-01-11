# USAGE
# python bright.py --image retina.png --radius 41
# python bright.py --image images/retina-noise.png --radius 41

# 导入必要的包
import math
import random
import numpy as np # 数值处理
import argparse # 命令行参数
import cv2 #绑定openCV
import matplotlib.pyplot as plt # plt 用于显示图片

# 构建命令行参数并解析
def raw_mask_generation(image,radius=21,num_raws=40):
    gt = image.copy()
    image = np.transpose(image,[1,2,0])
# 加载图像，复制图像并转换为灰度图
# image = cv2.imread('D:\code-0620/DRIVE/test/images/02_test.tif')#args["image"])
    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(orig,dtype=np.float64)
    # 查找图像中最亮点的敏感方法是使用cv2.minMaxLoc，称其敏感的原因是该方法极易受噪音干扰，可以通过预处理步骤应用高斯模糊解决。
    # 寻找最小、最大像素强度所在的（x,y）
    # (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    # # 在最大像素上绘制空心蓝色圆圈
    # cv2.circle(image, maxLoc, radius, (255, 0, 0), 2)

    # plt.subplot(221)
    # plt.imshow(image)
    # 展示该方法的结果
    # cv2.imshow("Naive", image)

    # 使用cv2.minMaxLoc，如果不进行任何预处理，可能会非常容易受到噪音干扰。
    # 相反，最好先对图像应用高斯模糊以去除高频噪声。这样，即使像素值非常大（同样由于噪声）也将被其邻居平均。
    # 在图像上应用高斯模糊消除高频噪声，然后寻找最亮的像素
    # 高斯模糊的半径取决于实际应用和要解决的问题；

    # print(minVal, maxVal, minLoc, center_point)
    gray = cv2.GaussianBlur(gray, (radius, radius), 0)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    center_point=maxLoc
    image = orig.copy()
    # cv2.circle(image, maxLoc, radius, (0, 0, 0), -1)
    # cv2.circle(mask, maxLoc, radius, (255, 0, 0), -1)

    # 展示效果显著提升后的方法结果
    # cv2.imshow("Robust", image)
    # cv2.waitKey(0)
    # num_raws=20
    point_x=np.zeros([num_raws*4])
    point_y=np.zeros([num_raws*4])
    length=len(image)
    # print(length)
    zeros=np.zeros(num_raws)
    ones=np.ones(num_raws)*length
    points=np.random.randint(0,length,num_raws)
    point_x[0:num_raws]=points
    # print(point_x,point_y)
    points=np.random.randint(0,length,num_raws)
    point_x[num_raws:num_raws*2]=points
    point_y[num_raws:num_raws*2]=ones
    # print(point_x,point_y)
    points=np.random.randint(0,length,num_raws)
    point_y[num_raws*2:num_raws*3]=points
    # print(point_x,point_y)
    points=np.random.randint(0,length,num_raws)
    point_y[num_raws*3:num_raws*4]=points
    point_x[num_raws*3:num_raws*4]=ones

    # print(point_x,point_y)

    for i in range(num_raws*4):
        # print((point_x[i],point_y[i]),i)
        cv2.line(image,(int(point_x[i]),int(point_y[i])),center_point,(0, 0, 0),4)
        cv2.line(mask,(int(point_x[i]),int(point_y[i])),center_point,(255, 0, 0),4)

    new_mask=np.ones_like(orig,dtype=np.int32)
    new_mask[mask>0]=0
    # plt.subplot(222)
    # # plt.plot([point_x,453],[point_y,272])
    # plt.imshow(image)
    # # plt.show()
    # # plt.subplot(222)
    # # plt.imshow(mm.cpu().data)#, cmap='Greys_r') # 显示图片
    # plt.subplot(223)
    # plt.imshow(mask)#, cmap='Greys_r') # 显示图片
    # plt.axis('off') # 不显示坐标轴

    #画线算法，参数为（原图像，点1，点2，画线数值）
    
    # image = np.transpose(image,[2,0,1])
    new_image = gt.copy()*new_mask[:,:,0]
    # new_image = np.transpose(new_image,[2,0,1])
    new_mask = np.transpose(new_mask,[2,0,1])
    # plt.subplot(224)
    # plt.imshow(np.transpose(new_image,[1,2,0]))#, cmap='Greys_r') # 显示图片
    
    # plt.show()
    return new_image, new_mask

def raw_mask_print(image,radius=21,num_raws=20,path=None):
    gt = image.copy()
    image = np.transpose(image,[1,2,0])
# 加载图像，复制图像并转换为灰度图
# image = cv2.imread('D:\code-0620/DRIVE/test/images/02_test.tif')#args["image"])
    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(orig,dtype=np.float64)
    # 查找图像中最亮点的敏感方法是使用cv2.minMaxLoc，称其敏感的原因是该方法极易受噪音干扰，可以通过预处理步骤应用高斯模糊解决。
    # 寻找最小、最大像素强度所在的（x,y）
    # (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    # # 在最大像素上绘制空心蓝色圆圈
    # cv2.circle(image, maxLoc, radius, (255, 0, 0), 2)

    plt.subplot(221)
    plt.imshow(image)
    # 展示该方法的结果
    # cv2.imshow("Naive", image)

    # 使用cv2.minMaxLoc，如果不进行任何预处理，可能会非常容易受到噪音干扰。
    # 相反，最好先对图像应用高斯模糊以去除高频噪声。这样，即使像素值非常大（同样由于噪声）也将被其邻居平均。
    # 在图像上应用高斯模糊消除高频噪声，然后寻找最亮的像素
    # 高斯模糊的半径取决于实际应用和要解决的问题；

    # print(minVal, maxVal, minLoc, center_point)
    gray = cv2.GaussianBlur(gray, (radius, radius), 0)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    center_point=maxLoc
    image = orig.copy()
    # cv2.circle(image, maxLoc, radius, (0, 0, 0), -1)
    # cv2.circle(mask, maxLoc, radius, (255, 0, 0), -1)

    # 展示效果显著提升后的方法结果
    # cv2.imshow("Robust", image)
    # cv2.waitKey(0)
    # num_raws=20
    point_x=np.zeros([num_raws*4])
    point_y=np.zeros([num_raws*4])
    length=len(image)
    # print(length)
    zeros=np.zeros(num_raws)
    ones=np.ones(num_raws)*length
    points=np.random.randint(0,length,num_raws)
    point_x[0:num_raws]=points
    # print(point_x,point_y)
    points=np.random.randint(0,length,num_raws)
    point_x[num_raws:num_raws*2]=points
    point_y[num_raws:num_raws*2]=ones
    # print(point_x,point_y)
    points=np.random.randint(0,length,num_raws)
    point_y[num_raws*2:num_raws*3]=points
    # print(point_x,point_y)
    points=np.random.randint(0,length,num_raws)
    point_y[num_raws*3:num_raws*4]=points
    point_x[num_raws*3:num_raws*4]=ones

    # print(point_x,point_y)

    for i in range(num_raws*4):
        # print((point_x[i],point_y[i]),i)
        cv2.line(image,(int(point_x[i]),int(point_y[i])),center_point,(0, 0, 0),4)
        cv2.line(mask,(int(point_x[i]),int(point_y[i])),center_point,(1, 0, 0),4)

    new_mask=np.ones_like(orig,dtype=np.int32)
    new_mask[mask>0]=0
    plt.subplot(222)
    # plt.plot([point_x,453],[point_y,272])
    plt.imshow(image)
    # plt.show()
    # plt.subplot(222)
    # plt.imshow(mm.cpu().data)#, cmap='Greys_r') # 显示图片
    plt.subplot(223)
    plt.imshow(mask)#, cmap='Greys_r') # 显示图片
    plt.axis('off') # 不显示坐标轴

    #画线算法，参数为（原图像，点1，点2，画线数值）
    
    # image = np.transpose(image,[2,0,1])
    new_image = gt.copy()*new_mask[:,:,0]
    # new_image = np.transpose(new_image,[2,0,1])
    new_mask = np.transpose(new_mask,[2,0,1])
    plt.subplot(224)
    plt.imshow(np.transpose(new_image,[1,2,0]))#, cmap='Greys_r') # 显示图片
    
    # plt.show()
    # print(mask.shape,image.shape)
    cv2.circle(image, maxLoc, radius, (0, 0, 0), -1)
    cv2.circle(mask, maxLoc, radius, (1, 0, 0), -1)
    plt.imsave(path.replace('images','masks'),mask)
    plt.imsave(path.replace('images','masked_images'),image)
    return new_image, new_mask