#
#   Lightnet dataset that works with brambox annotations
#   Copyright EAVISE
#

import os
import copy
import random
import logging
from PIL import Image
import torch
import torch.multiprocessing as multiprocessing
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import BatchSampler as torchBatchSampler
from torch.utils.data.dataloader import DataLoader as torchDataLoader
from torch.utils.data.dataloader import default_collate
import brambox.boxes as bbb


__all__ = ['BramboxData', 'DataLoader', 'list_collate']
log = logging.getLogger(__name__)


class BramboxData(Dataset):
    """Dataset for any brambox parsable annotation format.

    Args:
        anno_format (brambox.boxes.format): Annotation format
        anno_filename (list or str): Annotation filename, list of filenames or expandable sequence
        input_dimension (tuple): Tuple containing width,height values
        class_label_map (list): List of class_labels
        identify (function, optional): Lambda/function to get image based of annotation filename or image id; Default **replace/add .png extension to filename/id**
        img_transform (torchvision.transforms.Compose): Transforms to perform on the images
        anno_transform (torchvision.transforms.Compose): Transforms to perform on the annotations
        kwargs (dict): Keyword arguments that are passed to the brambox parser
    """

    def __init__(self, anno_format, anno_filename, input_dimension, class_label_map=None, identify=None, img_transform=None, anno_transform=None, **kwargs):
        super(BramboxData, self).__init__()
        self.__input_dim = input_dimension[:2]
        self.img_tf = img_transform
        self.anno_tf = anno_transform
        if callable(identify):
            self.id = identify
        else:
            self.id = lambda name : os.path.splitext(name)[0] + '.png'

        # Get annotations
        self.annos = bbb.parse(anno_format, anno_filename, identify=lambda f: f, class_label_map=class_label_map, **kwargs)
        self.keys = list(self.annos)

        # Add class_ids
        if class_label_map is None:
            log.warn(f'No class_label_map given, annotations wont have a class_id values for eg. loss function')  # NOQA
        for k, annos in self.annos.items():
            for a in annos:
                if class_label_map is not None:
                    try:
                        a.class_id = class_label_map.index(a.class_label)
                    except ValueError:
                        log.error(f'{a.class_label} is not found in the class_label_map')
                        raise
                else:
                    a.class_id = 0

        log.info(f'Dataset loaded: {len(self.keys)} images')

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        """Get (img, anno) tuple based of index from self.keys."""
        if not isinstance(index, int):
            self._input_dim = index[0]
            index = index[1]
        if index >= len(self):
            log.error(f'list index out of range [{index}/{len(self)-1}]')
            raise IndexError

        # Load
        img = Image.open(self.id(self.keys[index]))
        anno = copy.deepcopy(self.annos[self.keys[index]])

        # Transform
        if self.img_tf is not None:
            img = self.img_tf(img)
        if self.anno_tf is not None:
            anno = self.anno_tf(anno)

        if hasattr(self, '_input_dim'):
            del self._input_dim
        return img, anno

    @property
    def input_dim(self):
        """The dimensions that can be used by transforms to set the correct image size, etc.

        This allows transforms to have a single source of truth for the input dimension of the network.

        Return:
            list: Tuple containing the current width,height
        """
        if hasattr(self, '_input_dim'):
            return self._input_dim
        return self.__input_dim


class DataLoader(torchDataLoader):
    """Lightnet dataloader that enables on the fly resizing of the images.

    See :class:`torch.utils.data.DataLoader` for more information on the arguments.

    Note:
        This dataloader only works with :class:`lightnet.data.BramboxData` based datasets.
    """

    def __init__(self, *args, **kwargs):
        super(DataLoader, self).__init__(*args, **kwargs)
        shuffle = False
        sampler = None
        batch_sampler = None
        if len(args) > 5:
            shuffle = args[2]
            sampler = args[3]
            batch_sampler = args[4]
        elif len(args) > 4:
            shuffle = args[2]
            sampler = args[3]
            if 'batch_sampler' in kwargs:
                batch_sampler = kwargs['batch_sampler']
        elif len(args) > 3:
            shuffle = args[2]
            if 'sampler' in kwargs:
                sampler = kwargs['sampler']
            if 'batch_sampler' in kwargs:
                batch_sampler = kwargs['batch_sampler']
        else:
            if 'shuffle' in kwargs:
                shuffle = kwargs['shuffle']
            if 'sampler' in kwargs:
                sampler = kwargs['sampler']
            if 'batch_sampler' in kwargs:
                batch_sampler = kwargs['batch_sampler']

        # Use custom BatchSampler
        if batch_sampler is None:
            if sampler is None:
                if shuffle:
                    sampler = torch.utils.data.sampler.RandomSampler(self.dataset)
                else:
                    sampler = torch.utils.data.sampler.SequentialSampler(self.dataset)
            batch_sampler = BatchSampler(sampler, self.batch_size, self.drop_last, input_dimension=self.dataset.input_dim)

        self.sampler = sampler
        self.batch_sampler = batch_sampler

    def change_input_dim(self, value=32, randomize=True):
        """The function will compute a new size and update it on the next mini_batch.

        Args:
            value (int or tuple, optional): if ``random`` is false this value will be chosen for the new size, else this number represents a multiple for the random size; Default **32**
            randomize (boolean, optional): Whether to randomly compute a new size or set the size given; Default **True**
        """
        if not randomize:
            if isinstance(value, int):
                value = (value, value)
            else:
                value = (value[0], value[1])
            self.batch_sampler.new_input_dim = value
        else:
            if isinstance(value, int):
                size = (random.randint(0, 9) + 10) * value
                size = (size, size)
            else:
                size = ((random.randint(0, 9) + 10) * value[0], (random.randint(0, 9) + 10) * value[1])
            self.batch_sampler.new_input_dim = size


class BatchSampler(torchBatchSampler):
    """This batch sampler will generate mini-batches of (dim, index) tuples from another sampler.

    It works just like the :class:`torch.utils.data.sampler.BatchSampler`, but it will prepend a dimension,
    whilst ensuring it stays the same across one mini-batch.
    """

    def __init__(self, *args, input_dimension=None, **kwargs):
        super(BatchSampler, self).__init__(*args, **kwargs)
        self.input_dim = input_dimension
        self.new_input_dim = None

    def __iter__(self):
        self.__set_input_dim()
        for batch in super(BatchSampler, self).__iter__():
            yield [(self.input_dim, idx) for idx in batch]
            self.__set_input_dim()

    def __set_input_dim(self):
        """The function randomly changes the the input dimension of the dataset."""
        if self.new_input_dim is not None:
            log.info(f'Resizing network {self.new_input_dim[:2]}')
            self.input_dim = (self.new_input_dim[0], self.new_input_dim[1])
            self.new_input_dim = None


def list_collate(batch):
    """Function that collates lists of items together into one list (of lists).

    Use this as the collate function in a Dataloader, if you want to have a list of items as an output, as opposed to tensors (eg. Brambox.boxes).
    """
    items = list(zip(*batch))

    for i in range(len(items)):
        if isinstance(items[i][0], list):
            items[i] = list(items[i])
        else:
            items[i] = default_collate(items[i])

    return items
