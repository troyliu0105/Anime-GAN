import os
import cv2 as cv
import numpy as np
import tqdm
import argparse

argp = argparse.ArgumentParser(description="This script is used to crop and resize as specific size")
argp.add_argument('--path', type=str, required=True, help='Path of original images')
argp.add_argument('--dest', type=str, required=True, help='Path of cropped images')
argp.add_argument('--size', type=int, default=512, help='size of desire')
argp.add_argument('--delete_old', type=bool, default=False, help='whether delete original images')
opt = argp.parse_args()


def convert_to_resized(path, save_path, sepect_size=None, delete_old=False):
    img_lists = [item.path for item in os.scandir(path) if item.name.lower().endswith('jpg')]
    for img_path in tqdm.tqdm(img_lists, desc='Converting...',
                              leave=False,
                              dynamic_ncols=True,
                              unit='imgs',
                              unit_scale=True):
        img = cv.imread(img_path)
        h, w, _ = img.shape

        origin_name = os.path.split(img_path)[1]
        cropped_img_path = os.path.join(save_path, origin_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 截取竖型图片的上半部分
        if h > w:
            size = (w, w)
            r = (h - w) // 5
            if r == 0:
                r = 1
            anchor_y = np.random.choice(r, size=1)[0]
            cropped_img = img[anchor_y:anchor_y + size[0], :, :]
        else:
            size = (h, h)
            anchor_x = (w - h) // 2
            cropped_img = img[:, anchor_x: anchor_x + size[1]]
        if sepect_size:
            cropped_img = cv.resize(cropped_img, sepect_size)

        cv.imwrite(cropped_img_path, cropped_img)
        if delete_old:
            os.remove(img_path)


def get_face(img_path, save_path, cascade: cv.CascadeClassifier):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    p, n = os.path.split(img_path)
    img = cv.imread(img_path)

    # when call imwrite, cv2 will treat the arg as BGR, so... I won't change color here
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(64, 64))
    if len(faces) > 0:
        face_id = 0
        for (x, y, w, h) in faces:
            save_path = os.path.join(save_path, 'face_{}'.format(face_id) + n)
            face = img[y:y + h, x:x + w]
            cv.imwrite(save_path, face)
            face_id += 1


if __name__ == '__main__':
    # noinspection PyUnresolvedReferences
    convert_to_resized(path=opt.path,
                       save_path=opt.dest,
                       sepect_size=(opt.size, opt.size),
                       delete_old=opt.delete_old)
