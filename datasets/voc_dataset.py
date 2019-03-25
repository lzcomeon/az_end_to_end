import os
import numpy as np
from .util import read_image
import xml.etree.ElementTree as ET



VOC_BBOX_LABEL_NAMES = (
    'background',
    'tomato',
)


class VOCBboxDataset:
    """Bounding box dataset for PASCAL `VOC`_.

        .. _`VOC`: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

        The index corresponds to each image.

        When queried by an index, if :obj:`return_difficult == False`,
        this dataset returns a corresponding
        :obj:`img, bbox, label`, a tuple of an image, bounding boxes and labels.
        This is the default behaviour.
        If :obj:`return_difficult == True`, this dataset returns corresponding
        :obj:`img, bbox, label, difficult`. :obj:`difficult` is a boolean array
        that indicates whether bounding boxes are labeled as difficult or not.

        The bounding boxes are packed into a two dimensional tensor of shape
        :math:`(R, 4)`, where :math:`R` is the number of bounding boxes in
        the image. The second axis represents attributes of the bounding box.
        They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`, where the
        four attributes are coordinates of the top left and the bottom right
        vertices.

        The labels are packed into a one dimensional tensor of shape :math:`(R,)`.
        :math:`R` is the number of bounding boxes in the image.
        The class name of the label :math:`l` is :math:`l` th element of
        :obj:`VOC_BBOX_LABEL_NAMES`.

        The array :obj:`difficult` is a one dimensional boolean array of shape
        :math:`(R,)`. :math:`R` is the number of bounding boxes in the image.
        If :obj:`use_difficult` is :obj:`False`, this array is
        a boolean array with all :obj:`False`.

        The type of the image, the bounding boxes and the labels are as follows.

        * :obj:`img.dtype == numpy.float32`
        * :obj:`bbox.dtype == numpy.float32`
        * :obj:`label.dtype == numpy.int32`
        * :obj:`difficult.dtype == numpy.bool`

        Args:
            data_dir (string): Path to the root of the training data.
                i.e. "/data/image/voc/VOCdevkit/VOC2007/"
            split ({'train', 'val', 'trainval', 'test'}): Select a split of the
                dataset. :obj:`test` split is only available for
                2007 dataset.
            year ({'2007', '2012'}): Use a dataset prepared for a challenge
                held in :obj:`year`.
            use_difficult (bool): If :obj:`True`, use images that are labeled as
                difficult in the original annotation.
            return_difficult (bool): If :obj:`True`, this dataset returns
                a boolean array
                that indicates whether bounding boxes are labeled as difficult
                or not. The default value is :obj:`False`.

        """
    def __init__(self, data_dir, split='trainval'):

        id_list_file = os.path.join(
            data_dir, 'ImageSets/Main/{0}.txt'.format(split))

        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.data_dir = data_dir
        self.label_names = VOC_BBOX_LABEL_NAMES

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        """Returns the i-th example.

                Returns a color image and bounding boxes. The image is in CHW format.
                The returned image is RGB.

                Args:
                    i (int): The index of the example.

                Returns:
                    tuple of an image and bounding boxes

        """
        id_ = self.ids[i]
        anno_path = os.path.join(self.data_dir, 'Annotations', id_ + '.xml')
        anno = ET.parse(anno_path)
        bbox = list()
        label = list()

        for obj in anno.findall('object'):
            bndbox_anno = obj.find('bndbox')
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('xmin', 'ymin', 'xmax', 'ymax')])
            name = obj.find('name').text.lower().strip()
            label.append(VOC_BBOX_LABEL_NAMES.index(name))
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)

        img_path = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        img = read_image(img_path)

        return img, bbox, label

    __getitem__ = get_example






