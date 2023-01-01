import csv
import cv2
import os

import numpy as np


train_set_dir = 'train_set'
test_set_dir = 'test_set'

if not os.path.exists(train_set_dir):
    os.mkdir(train_set_dir)

if not os.path.exists(test_set_dir):
    os.mkdir(test_set_dir)



train_source_dir = 'GTSRB/Final_Training/Images'

for cls in range(43):

    if cls < 10:
        cls_dir = '0000%d' % cls
        csv_file = 'GT-0000%d.csv' % cls
    else:
        cls_dir = '000%d' % cls
        csv_file = 'GT-000%d.csv' % cls

    source_dir = os.path.join(train_source_dir,cls_dir)

    reader = csv.reader(
        open(os.path.join(source_dir, csv_file)), delimiter=";"
    )


    for rid,row in enumerate(reader):
        if rid == 0:
            continue
        Filename, Width, Height, X1, Y1, X2, Y2, ClassId = row
        X1 = int(X1)
        Y1 = int(Y1)
        X2 = int(X2)
        Y2 = int(Y2)
        ClassId = int(ClassId)

        img = cv2.imread(os.path.join(source_dir, Filename))
        sign = img[Y1:Y2,X1:X2]
        sign = cv2.resize(sign, (32, 32))

        save_dir = os.path.join(train_set_dir, cls_dir)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_file_path = os.path.join(save_dir,Filename)
        cv2.imwrite(save_file_path, sign)
        print('[train set] save : %s' % save_file_path)




test_source_dir = 'GTSRB-2/Final_Test/Images'
csv_file = 'GT-final_test.csv'
reader = csv.reader(
        open(os.path.join(test_source_dir, csv_file)), delimiter=";"
    )
for rid, row in enumerate(reader):
    if rid == 0:
        continue
    Filename, Width, Height, X1, Y1, X2, Y2, ClassId = row
    X1 = int(X1)
    Y1 = int(Y1)
    X2 = int(X2)
    Y2 = int(Y2)
    cls = int(ClassId)

    if cls < 10:
        cls_dir = '0000%d' % cls
    else:
        cls_dir = '000%d' % cls

    img = cv2.imread(os.path.join(test_source_dir, Filename))
    sign = img[Y1:Y2, X1:X2]
    sign = cv2.resize(sign, (32, 32))



    save_dir = os.path.join(test_set_dir, cls_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_file_path = os.path.join(save_dir, Filename)
    cv2.imwrite(save_file_path, sign)
    print('[test set] save : %s' % save_file_path)
