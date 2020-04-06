from __future__ import print_function, division
from __future__ import absolute_import

import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import random
import torch
import pdb

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:0
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, class_to_idx, extensions, CAM=False):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        if target == '-1' or target == '0':
            continue
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    if CAM :
                        cam = fname.split('c', 1)[1][0]
                        item = (path, class_to_idx[target], int(target), int(cam))
                    else:
                        if target == '-1_c':
                            item = (path, -1, -1)
                        else:
                            item = (path, class_to_idx[target], int(target))
                    images.append(item)

    return images

def make_triplet_dataset(dir, class_to_idx, extensions, CAM=False):
    rtn = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        images = []
        if target == '-1' or target == '0':
            continue
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    if CAM :
                        cam = fname.split('c', 1)[1][0]
                        item = (path, class_to_idx[target], int(target), int(cam))
                    else:
                        item = (path, class_to_idx[target], int(target))
                    images.append(item)
        rtn.append(images)
    return rtn


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def make_triplet(data):
    triplet_path = []
    classes_indices = torch.randperm(len(data))
    for i in classes_indices:
        class_index = i-1 # to adjust the num randperm generated
        image_num = len(data[class_index])
        images_indices = torch.randperm(image_num)
        for j in range(len(images_indices) - 1):
            anchor_index = images_indices[j] - 1
            positive_index = images_indices[j + 1] - 1
            negative_class = random.randint(0, len(data)-1)
            while negative_class == class_index:
                negative_class = random.randint(0, len(data)-1)
            negative_index = random.randint(0, len(data[negative_class])-1)
            item = (data[class_index][anchor_index][0], data[class_index][positive_index][0], data[negative_class][negative_index][0])
            triplet_path.append(item)
        anchor_index = images_indices[len(images_indices)-1]
        positive_index = images_indices[0]
        while negative_class == class_index:
            negative_class = random.randint(0, len(data)-1)
        negative_index = random.randint(0, len(data[negative_class])-1)
        item = (data[class_index][anchor_index][0], data[class_index][positive_index][0], data[negative_class][negative_index][0])
        triplet_path.append(item)

    return triplet_path


class Dataset(data.Dataset):
    def __init__(self, image_dir, transform=None, transform_resize=None, CAM=False):
        self.transform = transform
        self.transform_resize = transform_resize
        classes, class_to_idx = find_classes(image_dir)
        self.data = make_dataset(image_dir, class_to_idx, IMG_EXTENSIONS, CAM)
        self.CAM = CAM

    def __getitem__(self, index):
        if self.CAM:
            path, pid, real_id, cam = self.data[index]
        else:
            path, pid, real_id = self.data[index]
        img = default_loader(path)
        if self.transform is not None:
            img_normal = self.transform(img)
        if self.transform_resize is not None:
            img_resize = self.transform_resize(img)
        if self.CAM:
            return img_normal, pid, real_id, cam
        else:
            if self.transform_resize is not None:
                return img_normal, img_resize, pid, real_id
            else:
                return img_normal, pid, real_id

    def __len__(self):
        return len(self.data)

class MSMT17(data.Dataset):
    def __init__(self, image_dir, transform=None, transform_resize=None):
        self.transform = transform
        classes, class_to_idx = find_classes(image_dir)
        self.data = make_dataset(image_dir, class_to_idx, IMG_EXTENSIONS, False)

    def __getitem__(self, index):
        path, pid, real_id = self.data[index]
        img = default_loader(path)
        img = self.transform(img)
        return img, pid, real_id

    def __len__(self):
        return len(self.data)

class DatasetTri(data.Dataset):
    def __init__(self, image_dir, transform=None, transform_resize=None, CAM=False):
        self.transform = transform
        self.transform_resize = transform_resize
        classes, class_to_idx = find_classes(image_dir)
        self.data = make_dataset(image_dir, class_to_idx, IMG_EXTENSIONS, CAM)
        self.CAM = CAM

    def __getitem__(self, index):
        if self.CAM:
            path, pid, real_id, cam = self.data[index]
        else:
            path, pid, real_id = self.data[index]
        img = default_loader(path)
        rand_index = random.randint(0, len(self.data)-1)
        while rand_index == index:
            rand_index = random.randint(0, len(self.data)-1)
        img_negative_idx = rand_index
        img_negative_path, _, _ = self.data[img_negative_idx]
        img_negative = default_loader(img_negative_path)
        if self.transform is not None:
            img_normal = self.transform(img)
        if self.transform_resize is not None:
            img_negative = self.transform(img_negative)
            img_resize = self.transform_resize(img)
        if self.CAM:
            return img_normal, pid, real_id, cam
        else:
            if self.transform_resize is not None:
                return img_normal, img_resize, img_negative, pid, real_id
            else:
                return img_normal, pid, real_id

    def __len__(self):
        return len(self.data)

class DatasetMulti(data.Dataset):
    def __init__(self, image_dir, transform=None, transform_resize=None, CAM=False):
        self.transform = transform
        self.transform_resize = transform_resize
        classes, class_to_idx = find_classes(image_dir)
        self.data = make_dataset(image_dir, class_to_idx, IMG_EXTENSIONS, CAM)
        self.CAM = CAM

    def __getitem__(self, index):
        path, pid, real_id = self.data[index]
        img = default_loader(path)
        rand_index = random.randint(0, len(self.data)-1)
        while rand_index == index:
            rand_index = random.randint(0, len(self.data)-1)
        img_negative_idx = rand_index
        img_negative_path, _, _ = self.data[img_negative_idx]
        img_negative = default_loader(img_negative_path)
        if self.transform is not None:
            img_normal = self.transform(img)
        if self.transform_resize is not None:
            img_negative = self.transform(img_negative)
            img_resize = self.transform_resize(img)
        if self.CAM:
            return img_normal, pid, real_id, cam
        else:
            if self.transform_resize is not None:
                return img_normal, img_resize, img_negative, pid, real_id
            else:
                return img_normal, pid, real_id

    def __len__(self):
        return len(self.data)



class DatasetAug(data.Dataset):
    def __init__(self, image_dir, transform=None, transform2=None, transform3=None, transform4=None, CAM=False):
        self.transform = transform
        self.transform2 = transform2
        self.transform3 = transform3
        self.transform4 = transform4
        classes, class_to_idx = find_classes(image_dir)
        self.data = make_dataset(image_dir, class_to_idx, IMG_EXTENSIONS, CAM)
        self.CAM = CAM

    def __getitem__(self, index):
        if self.CAM:
            path, pid, real_id, cam = self.data[index]
        else:
            path, pid, real_id = self.data[index]
        img = default_loader(path)
        img1 = self.transform(img)
        img1_pos = self.transform2(img)
        img2 = self.transform3(img)
        img2_pos = self.transform4(img)
        return img1, img1_pos, img2, img2_pos, pid, real_id

    def __len__(self):
        return len(self.data)

class DatasetAugThree(data.Dataset):
    def __init__(self, image_dir, transform=None, transform2=None, transform3=None, transform4=None, CAM=False):
        self.transform = transform
        self.transform2 = transform2
        self.transform3 = transform3
        self.transform4 = transform4
        classes, class_to_idx = find_classes(image_dir)
        self.data = make_dataset(image_dir, class_to_idx, IMG_EXTENSIONS, CAM)
        self.CAM = CAM

    def __getitem__(self, index):
        if self.CAM:
            path, pid, real_id, cam = self.data[index]
        else:
            path, pid, real_id = self.data[index]
        img = default_loader(path)
        img_normal = self.transform(img)
        img2 = self.transform2(img)
        img3 = self.transform3(img)
        img4 = self.transform4(img)
        return img_normal, img2, img3, img4, pid, real_id

    def __len__(self):
        return len(self.data)



class DatasetTriphard(data.Dataset):
    def __init__(self, image_dir, transform=None, transform_resize=None, CAM=False):
        self.transform = transform
        self.transform_resize = transform_resize
        classes, class_to_idx = find_classes(image_dir)
        self.data = make_dataset(image_dir, class_to_idx, IMG_EXTENSIONS, CAM)
        self.CAM = CAM

    def __getitem__(self, index):
        if self.CAM:
            path, pid, real_id, cam = self.data[index]
        else:
            path, pid, real_id = self.data[index]
        img = default_loader(path)
        if self.transform is not None:
            img_normal = self.transform(img)
        if self.transform_resize is not None:
            img_resize = self.transform_resize(img)
        if self.CAM:
            return img_normal, pid, real_id, cam
        else:
            if self.transform_resize is not None:
                return img_normal, img_resize, pid, real_id
            else:
                return img_normal, pid, real_id

    def __len__(self):
        return len(self.data)

class DatasetTriplet(data.Dataset):
    def __init__(self, image_dir, transform=None, transform_resize=None, CAM=False):
        self.transform = transform
        self.transform_resize = transform_resize
        classes, class_to_idx = find_classes(image_dir)
        self.data = make_triplet_dataset(image_dir, class_to_idx, IMG_EXTENSIONS, CAM)
        self.triplet_path = make_triplet(self.data)

    def __getitem__(self, index):
         anchor_path, positive_path, negative_path = self.triplet_path[index]
         anchor = default_loader(anchor_path)
         positive = default_loader(positive_path)
         negative = default_loader(negative_path)
         anchor_img = self.transform(anchor)
         positive_img = self.transform(positive)
         negative_img = self.transform(negative)
         return anchor_img, positive_img, negative_img

    def __len__(self):
        return len(self.triplet_path)

    def Shuffle(self):
        self.triplet_path = make_triplet(self.data)
