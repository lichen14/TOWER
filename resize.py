import os.path
from PIL import Image
from glob import glob
from tqdm import tqdm
# img_file:图片的路径
# path_save:保存路径
# width：宽度
# height：长度
img_list = glob('./datasets/NIH Chest X-ray/image/*')
path_save = './datasets/NIH Chest X-ray/image224'
for img_file in tqdm(img_list,total = len(img_list)):
    img = Image.open(img_file)
    new_image = img.resize((224,224),Image.BILINEAR)
    new_image.save(os.path.join(path_save,os.path.basename(img_file)))