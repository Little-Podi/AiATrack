import torch

from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from lib.utils.merge import merge_feature_sequence
from lib.utils.misc import NestedTensor
from . import BaseActor


class AIATRACKActor(BaseActor):
    """
    Actor for training.
    """

    def __init__(self, net, objective, loss_weight, settings):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # Batch size

    def __call__(self, data):
        """
        Args:
            data: The input data, should contain the fields 'reference', 'search', 'gt_bbox'.
            reference_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)

        Returns:
            loss: The training loss.
            status: Dict containing detailed losses.
        """

        # Forward pass
        out_dict = self.forward_pass(data)

        # Compute losses
        # Process the groundtruth
        gt_bboxes = data['search_anno']  # (Ns, batch, 4) (x1,y1,w,h)
        loss, status = self.compute_losses(out_dict, gt_bboxes[0], data['proposal_iou'])
        return loss, status

    def forward_pass(self, data):
        # Process the search regions (t-th frame)
        search_dict_list = list()
        search_img = data['search_images'].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
        search_att = data['search_att'].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)
        search_dict_list.append(self.net(img=NestedTensor(search_img, search_att), mode='backbone'))
        search_dict = merge_feature_sequence(search_dict_list)

        # Process the reference frames
        feat_dict_list = list()
        refer_reg_list = list()
        for i in range(data['reference_images'].shape[0]):
            reference_dict_list = list()
            reference_img_i = data['reference_images'][i].view(-1, *data['reference_images'].shape[
                                                                    2:])  # (batch, 3, 320, 320)
            reference_att_i = data['reference_att'][i].view(-1, *data['reference_att'].shape[2:])  # (batch, 320, 320)
            reference_dict_list.append(self.net(img=NestedTensor(reference_img_i, reference_att_i), mode='backbone'))
            feat_dict_list.append(merge_feature_sequence(reference_dict_list))
            refer_reg_list.append(data['reference_region'][i])

        # Run the transformer and compute losses
        out_embed, _, _, _ = self.net(search_dic=search_dict, refer_dic_list=feat_dict_list,
                                      refer_reg_list=refer_reg_list, mode='transformer')

        # Forward the corner head
        out_dict = self.net(out_embed=out_embed, proposals=data['search_proposals'],
                            mode='heads')  # out_dict: (B, N, C), outputs_coord: (1, B, N, C)
        return out_dict

    def compute_losses(self, pred_dict, gt_bbox, iou_gt, return_status=True):
        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError('ERROR: network outputs is NaN! stop training')
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # Compute GIoU and IoU
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # Compute L1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)

        iou_pred = pred_dict['pred_iou']
        iou_loss = self.objective['iou'](iou_pred, iou_gt)

        # Weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight[
            'iou'] * iou_loss
        if return_status:
            # Status for log
            mean_iou = iou.detach().mean()
            status = {'Ls/total': loss.item(),
                      'Ls/giou': giou_loss.item(),
                      'Ls/l1': l1_loss.item(),
                      'Ls/iou': iou_loss.item(),
                      'IoU': mean_iou.item()}
            return loss, status
        else:
            return loss
