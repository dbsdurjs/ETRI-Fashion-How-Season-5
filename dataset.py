import torch.utils.data
import numpy as np
from torchvision import transforms
from skimage import io, transform, color
import timm
import cv2, random, os

class BackGround(object):
    """Operator that resizes to the desired size while maintaining the ratio
            fills the remaining part with a black background

        Args:
            output_size (tuple or int): Desired output size. If tuple, output is
                matched to output_size.
    """
    
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, image, landmarks, sub_landmarks=None):
        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size, self.output_size * w / h
            else:
                new_h, new_w = self.output_size * h / w, self.output_size
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w), mode='constant')

        if landmarks is not None:
            landmarks = landmarks * [new_w / w, new_h / h]

            new_image = np.zeros((self.output_size, self.output_size, 3))

            if h > w:
                new_image[:,(112 - new_w//2):(112 - new_w//2 + new_w),:] = img
                landmarks = landmarks + [112 - new_w//2, 0]
            else:
                new_image[(112 - new_h//2):(112 - new_h//2 + new_h), :, :] = img
                landmarks = landmarks + [0, 112 - new_h//2]

            if sub_landmarks is not None:
                sub_landmarks = sub_landmarks * [new_w / w, new_h / h]
                if h > w:
                    sub_landmarks = sub_landmarks + [112 - new_w // 2, 0]
                else:
                    sub_landmarks = sub_landmarks + [0, 112 - new_h // 2]
                return new_image, landmarks, sub_landmarks
            else:
                return new_image, landmarks
        else:
            new_image = np.zeros((self.output_size, self.output_size, 3))
            if h > w:
                new_image[:,(112 - new_w//2):(112 - new_w//2 + new_w),:] = img
            else:
                new_image[(112 - new_h//2):(112 - new_h//2 + new_h), :, :] = img

            return new_image


class BBoxCrop(object):
    """ Operator that crops according to the given bounding box coordinates. """
    
    def __call__(self, image, x_1, y_1, x_2, y_2):
        h, w = image.shape[:2]

        top = y_1
        left = x_1
        new_h = y_2 - y_1
        new_w = x_2 - x_1

        image = image[top: top + new_h,
                      left: left + new_w]

        return image

class ETRIDataset_color(torch.utils.data.Dataset):
    """ Dataset containing color category. """
    
    def __init__(self, df, base_path, mode='train'):
        self.df = df
        self.base_path = base_path
        self.bbox_crop = BBoxCrop()
        self.background = BackGround(224)
        self.mode = mode
        
        # add
        if mode == 'train':
            self.transform_image = transforms.Compose([
                transforms.CenterCrop(200),
                transforms.Resize((224,224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(contrast=0.2),
                transforms.RandomAffine(degrees=30, scale=(0.6, 1.4)),
                transforms.ToTensor(),
            ])
        else:
            self.transform_image = transforms.Compose([
                transforms.CenterCrop(200),
                transforms.Resize((224,224)),
                transforms.ToTensor(),
            ])
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([.5, .5, .5], [.5, .5, .5])
        
        self.to_pil = transforms.ToPILImage()
        
        
        self.label_freq = {_class: 0 for _class in self.df['Color'].unique()}
        self.targets = [_class for _class in self.df['Color']]
        
        for _class in self.targets:
            self.label_freq[_class] += 1


    def __getitem__(self, i):
        sample = self.df.iloc[i]
        image = io.imread(self.base_path + sample['image_name'])
        if image.shape[2] != 3:
            image = color.rgba2rgb(image)
        color_label = sample['Color']
        
        # crop only if bbox info is available
        try:
            bbox_xmin = sample['BBox_xmin']
            bbox_ymin = sample['BBox_ymin']
            bbox_xmax = sample['BBox_xmax']
            bbox_ymax = sample['BBox_ymax']
    
            image = self.bbox_crop(image, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax)
        except:
            pass
        
        image = self.background(image, None) # 224,224,3
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        if self.mode == 'train':
            image = transforms.ToPILImage()(image)
            image = self.transform_image(image)
            
            image = np.array(image) # 3,224,224
            
            if image.ndim == 3 and image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))

                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)  # 224,224,3

        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        image_hsv = self.to_tensor(image_hsv)
        image_hsv = self.normalize(image_hsv)
        
        image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        image_lab = self.to_tensor(image_lab)
        image_lab = self.normalize(image_lab)
        
        image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        image_hls = self.to_tensor(image_hls)
        image_hls = self.normalize(image_hls)
        
        image_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        image_yuv = self.to_tensor(image_yuv)
        image_yuv = self.normalize(image_yuv)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb = self.to_tensor(image_rgb)
        image_rgb = self.normalize(image_rgb)

        
        ret = {}
        ret['ori_image'] = image
        ret['image_hsv'] = image_hsv
        ret['image_lab'] = image_lab
        ret['image_hls'] = image_hls
        ret['image_yuv'] = image_yuv
        ret['image_rgb'] = image_rgb
        ret['color_label'] = color_label

        return ret

    def __len__(self):
        return len(self.df)
    
    def get_labels(self):
        return self.targets