import torch
import random
import numpy as np

class intensity_shift(object):

    def __call__(self, inputs):
        image = inputs['image']
        label = inputs['label']

        for i in range(image.shape[0]):
            shift = random.uniform(-0.1, 0.1)
            scale = random.uniform(0.9, 1.1)
            img = image[i, ...]
            image[i, ...] = scale * (img + shift)

        return {"image": image, "label": label}

class mirror_flip(object):

    def __call__(self, inputs):
        image = inputs['image']
        label = inputs['label']

        for i in range(1, image.ndim):
            if random.uniform(0, 1) < 0.5:
                image = np.flip(image, axis=i)
                label = np.flip(label, axis=i-1)

        return {"image": image.copy(), "label": label.copy()}

class image_normalization(object):

    def __call__(self, inputs):
        image = inputs['image']
        label = inputs['label']

        for i in range(image.shape[0]):
            for j in range(image.shape[3]):
                image_slice = image[i,:,:,j]

                #zero_mean
                image_slice = image_slice - np.mean(image_slice)
                #unit_stdev (based on non-zero voxels only)
                if np.std(image_slice) != 0:
                    image_slice = image_slice / np.std(image_slice)

                image[i, :, :, j] = image_slice

        return {"image": image, "label": label}

class random_crop(object):

    def __init__(self, size) -> None:
        super().__init__()
        self.size = size

    def __call__(self,inputs):
        image = inputs['image']
        label = inputs['label']

        x = random.randint(0, 240-self.size[0])
        y = random.randint(0, 240-self.size[1])
        z = random.randint(0, 155-self.size[2])

        image = image[:, x:x+self.size[0], y:y+self.size[1], z:z+self.size[2]]
        label = label[x:x+self.size[0], y:y+self.size[1], z:z+self.size[2]]
        label = np.where(label == 4, 3, label)

        return {"image": image, "label": label}

class center_crop(object):

    def __call__(self, inputs):
        image = inputs['image']
        label = inputs['label']

        image = image [:, 40:200, 24:216, 12:140]
        label = label [40:200, 24:216, 12:140]
        label = np.where(label == 4, 3, label)

        return {"image": image, "label": label}


class ToTensor(object):

    def __call__(self, inputs):
        image = inputs['image']
        label = inputs['label']

        image = torch.from_numpy(image)
        label = torch.from_numpy(label)

        return {"image": image, "label": label}

class one_hot_encoding(object):

    def __call__(self, inputs):
        image = inputs['image']
        label = inputs['label']

        label = torch.nn.functional.one_hot(label.to(torch.int64), num_classes=4)
        label = torch.movedim(label, 3, 0)
        label = label[1:,:,:,:]

        return {"image": image, "label": label}
