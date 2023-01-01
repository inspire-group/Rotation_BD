
import csv
import cv2
import os

import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-num', type=int)
args = parser.parse_args()

dir_path = "FullIJCNN2013"
gt_file = "gt.txt"
reader = csv.reader(
    open(os.path.join(dir_path,gt_file)), delimiter=";"
)


angle = args.num

save_root_dir = 'poison_samples'
sub_dir = 'rotation_nc%d' % angle
if not os.path.exists(save_root_dir):
    os.makedirs(save_root_dir)
if not os.path.exists(os.path.join(save_root_dir, sub_dir)):
    os.makedirs(os.path.join(save_root_dir, sub_dir))



num = 0
for row in reader:

    img_name, x1, y1, x2, y2, label = row
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)
    label = int(label)

    # (x1,y1) : left_upper corner
    # (x2,y2) : right_lower corner

    border = np.array( [
        [x1,x2],
        [y1,y2],
        [1,1],
    ] )

    img = cv2.imread(os.path.join(dir_path, img_name))
    sign_center = ( (x1+x2)//2, (y1+y2)//2 )
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D(sign_center, angle, 1)
    img_trans = cv2.warpAffine(img, M, (cols,rows))
    #img_trans = cv2.rectangle(img_trans, (border[0][0], border[1][0]), (border[0][1], border[1][1]), (255, 0, 0), 2)
    sign = img_trans[y1:y2,x1:x2]
    sign = cv2.resize(sign,(32,32))
    sticker_path = "dfresult/NC/rb"+str(angle)+"aug0pr0003/"+"gtsrb_visualize_mask_label_0.png"
    sticker = cv2.imread(sticker_path)
    print(sticker.shape,sign.shape)
    
    #cv2.imshow('img',sign)
    #cv2.waitKey(0)

    if label<10:
        class_dir = '0000%d' % label
    else:
        class_dir = '000%d' % label
        
    if not os.path.exists(os.path.join(save_root_dir,sub_dir,class_dir)):
        os.mkdir(os.path.join(save_root_dir,sub_dir,class_dir))

    file_name = '%d.png' % num
    cv2.imwrite(os.path.join(save_root_dir,sub_dir,class_dir,file_name), sign)

    print('[num:%d] Save %s' % (num,os.path.join(save_root_dir,sub_dir,class_dir,file_name)))
    num += 1




