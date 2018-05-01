import csv
import os
import shutil

dir = "C:/Users/neroo/OneDrive/Desktop/BigData/rtsd-r1.tar/rtsd-r1/train"


def create_directory(dir_name):
    if os.path.exists(dir + '/' + dir_name):
        shutil.rmtree(dir + '/' + dir_name)
    os.makedirs(dir + '/' + dir_name)


def copy_images(filename, class_number):
    shutil.copy2(os.path.join('C:/Users/neroo/OneDrive/Desktop/BigData/rtsd-r1.tar/rtsd-r1/train', filename), os.path.join(dir, class_number))

with open('C:/Users/neroo/OneDrive/Desktop/BigData/rtsd-r1.tar/rtsd-r1/gt_train.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        print('Copying ' + row['filename'] + ' image..')
        copy_images(row['filename'], row['class_number'])
        # create_directory(row['class_number'])