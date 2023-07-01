import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import lib.train.data.processing_utils as prutils
from lib.utils import TensorDict


def stack_tensors(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x


class BaseProcessing:
    """
    Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it
    through the network. For example, it can be used to crop a search region around the object, apply various data
    augmentations, etc.
    """

    def __init__(self, transform=transforms.ToTensor(), search_transform=None, joint_transform=None):
        """
        Args:
            transform: The set of transformations to be applied on the images.
                       Used only if search_transform is None.
            search_transform: The set of transformations to be applied on the search images.
                              If None, the 'transform' argument is used instead.
            joint_transform: The set of transformations to be applied 'jointly' on the reference and search images.
                             For example, it can be used to convert both reference and search images to grayscale.
        """

        self.transform = {'search': transform if search_transform is None else search_transform,
                          'reference': transform if search_transform is None else search_transform,
                          'joint': joint_transform}

    def __call__(self, data: TensorDict):
        raise NotImplementedError


class AIATRACKProcessing(BaseProcessing):
    """
    The processing class used for training LittleBoy. The images are processed in the following way.

    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region)
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz.
    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor,
                 mode='pair', settings=None, *args, **kwargs):
        """
        Args:
            search_area_factor: The size of the search region  relative to the target size.
            output_sz: An integer, denoting the size to which the search region is resized.
                       The search region is always square.
            center_jitter_factor: A dict containing the amount of jittering to be applied to the target center before
                                  extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor: A dict containing the amount of jittering to be applied to the target size before
                                 extracting the search region. See _get_jittered_box for how the jittering is done.
            mode: Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames.
        """

        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode
        self.settings = settings

    def _get_jittered_box(self, box, mode):
        """
        Jitter the input box.

        Args:
            box: Input bounding box.
            mode: String 'reference' or 'search' indicating reference or search data.

        Returns:
            torch.Tensor: Jittered box.
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)
        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def _generate_proposals(self, box):
        """
        Generates proposals by adding noise to the input box.

        Args:
            box: Input box.

        Returns:
            torch.Tensor: Array of shape (num_proposals, 4) containing proposals.
            torch.Tensor: Array of shape (num_proposals,) containing IoU overlap of each proposal with the input box.
                          The IoU is mapped to [-1, 1].
        """

        # Generate proposals
        num_proposals = 16

        proposals = torch.zeros((num_proposals, 4))
        gt_iou = torch.zeros(num_proposals)
        for i in range(num_proposals):
            proposals[i, :], gt_iou[i] = prutils.perturb_box(box, min_iou=0.1,
                                                             sigma_factor=[0.03, 0.05, 0.1, 0.2, 0.3])

        # Map to [-1, 1]
        gt_iou = gt_iou * 2 - 1
        return proposals, gt_iou

    def __call__(self, data: TensorDict):
        """
        Args:
            data: The input data, should contain the following fields:
                  'reference_images', search_images', 'reference_anno', 'search_anno'

        Returns:
            TensorDict: Output data block with following fields:
                        'reference_images', 'search_images', 'reference_anno', 'search_anno', 'test_proposals', 'proposal_iou'
        """

        # Apply joint transforms
        if self.transform['joint'] is not None:
            data['search_images'], data['search_anno'] = self.transform['joint'](image=data['search_images'],
                                                                                 bbox=data['search_anno'])
            data['reference_images'], data['reference_anno'] = self.transform['joint'](image=data['reference_images'],
                                                                                       bbox=data['reference_anno'],
                                                                                       new_roll=False)

        for s in ['search', 'reference']:
            # Add a uniform noise to the center pos
            if s in ['reference']:
                jittered_anno = [self._get_jittered_box(data[s + '_anno'][0], 'initial')]
                for a in data[s + '_anno'][1:]:
                    jittered_anno.append(self._get_jittered_box(a, s))
            else:
                jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # Check whether data is valid. Avoid too small bounding boxes
            w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]
            crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[s])
            if (crop_sz < 1).any():
                data['valid'] = False
                # print('too small box is found, replace it with new data')
                return data

            # Crop image region centered at jittered_anno box and get the attention mask
            crops, boxes, att_mask = prutils.jittered_center_crop(data[s + '_images'],
                                                                  jittered_anno,
                                                                  data[s + '_anno'],
                                                                  self.search_area_factor[s],
                                                                  self.output_sz[s])

            # Apply transforms
            data[s + '_images'], data[s + '_anno'], data[s + '_att'] = self.transform[s](
                image=crops, bbox=boxes, att=att_mask, joint=False)

            if s in ['reference']:
                feat_size = self.output_sz[s] // 16
                data[s + '_region'] = list()
                for anno in data[s + '_anno']:
                    target_region = torch.zeros((feat_size, feat_size))
                    x, y, w, h = (anno * feat_size).round().int()
                    target_region[max(y, 0):min(y + h, feat_size), max(x, 0):min(x + w, feat_size)] = 1
                    target_region = target_region.view(feat_size * feat_size, -1)
                    background_region = 1 - target_region
                    data[s + '_region'].append(torch.cat([target_region, background_region], dim=1))

            # Check whether elements in data[s + '_att'] is all 1
            # Which means all of elements are padded
            # Note that type of data[s + '_att'] is tuple, type of ele is torch.tensor
            for ele in data[s + '_att']:
                if (ele == 1).all():
                    data['valid'] = False
                    # print('values of original attention mask are all one, replace it with new data')
                    return data
            # More strict conditions: require the down-sampled masks not to be all 1
            for ele in data[s + '_att']:
                feat_size = self.output_sz[s] // 16  # 16 is the backbone stride
                # (1,1,128,128) (1,1,256,256) --> (1,1,8,8) (1,1,16,16)
                mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(torch.bool)[0]
                if (mask_down == 1).all():
                    data['valid'] = False
                    # print('values of down-sampled attention mask are all one, replace it with new data')
                    return data

        # Generate proposals
        iou_proposals, gt_iou = zip(*[self._generate_proposals(a) for a in data['search_anno']])
        data['search_proposals'] = list(iou_proposals)
        data['proposal_iou'] = list(gt_iou)

        data['valid'] = True

        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)
        return data
