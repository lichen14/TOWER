import torch
import torch.nn as nn

class SupConLoss(nn.Module):
    """Self-Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf, supported by the unsupervised contrastive loss in SimCLR"""
    def __init__(self, threshold=0.1, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, contrastive_method='simclr'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1) #cos in the last channel
        self.threshold = threshold
        self.contrastive_method = contrastive_method

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)->(N, 1,1, C)
        # y shape: (1, N, C)->(1, 1,N, C)
        # v shape: (N, N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].   #n_views is torch.cat~ed in train 
            labels: ground truth of shape [bsz]. #slice_position or partition
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:   # equal to SimCLR
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)    # return 2-D tensor with 1 on the diagonal and 0 elsewhere (n*n size)
        elif labels is not None:        # mask is None
            labels = labels.contiguous().view(-1, 1)  
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            if self.contrastive_method == 'gcl':
                mask = torch.eq(labels, labels.T).float().to(device)   
            elif self.contrastive_method == 'pcl':
                # return mask [bsz,bsz], where diagonal is 0 and elsewhere is |dis|, denote the distance of slices. and then compare with threshold for bool (T/F) and convert to float
                mask = (torch.abs(labels.T.repeat(batch_size,1) - labels.repeat(1,batch_size)) < self.threshold).float().to(device) 

        else:                # labels is None and mask is not None
            mask = mask.float().to(device)

        contrast_count = features.shape[1]  #feature n_views cat
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  #torch.unbind: Removes a tensor dimension. Returns a tuple of all slices along a given dimension
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits  [bsz*count,bsz*count]    
        logits = torch.div(
            self._cosine_simililarity(anchor_feature, contrast_feature),
            self.temperature)

        # tile mask [bsz,bsz] -> [bsz*count,bsz*count]  (i,i)=(i+bsz,i)=(i,i+bsz)=1
        mask = mask.repeat(anchor_count, contrast_count)    # in 'all' model, anchor_count = contrast_count = n_views
        # print(mask[0],mask[1],mask[18],mask[19])
        # mask-out self-contrast cases  return [bsz*count,bsz*count] with 0 on diagonal and 1 elsewhere
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device), # return (batch_size * anchor_count) *1 tensor
            0
        )
        mask = mask * logits_mask       #[bsz*count,bsz*count]. (i,i)=0 while (i,j)=original value => (i+bsz,i)=(i,i+bsz)=1 => positive pair  ##pixel-wise multi
        # print(mask)
        # compute log_prob  [bsz*count,bsz*count]
        exp_logits = torch.exp(logits) * logits_mask
        # print('logits.shape,logits_mask.shape,exp_logits.shape',logits.shape,logits_mask.shape,exp_logits.shape)

        # log_prob = mask *logits - torch.log(exp_logits.sum(1, keepdim=True))    # this is the Eq.(2) in PCL
        # # compute mean of log-likelihood over positive
        # mean_log_prob_pos = log_prob.sum(1) / mask.sum(1)#(mask * log_prob).sum(1) / mask.sum(1)
        
        #old
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))    # this is the Eq.(2) in PCL
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)#log_prob.sum(1) / mask.sum(1)#(mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        # print('loss',loss)
        return loss