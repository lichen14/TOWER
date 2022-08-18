import torch
from . import initialization as init

class SegmentationModel_1124(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x,y,m):
        """Sequentially pass `x` trough model`s encoder, decoder and heads 
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """
        
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        features_y = self.encoder(y)
        decoder_output_y = self.decoder(*features_y)

        masks = self.segmentation_head(decoder_output)
        
        # compute features   predictor is the mlp
        q1 = self.segmentation_head(decoder_output)#self.predictor(self.base_encoder(x1))
        q2 = self.segmentation_head(decoder_output_y)

        momentum_features_x = self.momentum_encoder(x)
        # q1 = self.momentum_decoder(*momentum_features_x)
        momentum_features_y = self.momentum_encoder(y)
        # q2 = self.momentum_decoder(*momentum_features_y)
        # print(momentum_features_x.shape,momentum_features_y.shape)
        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        with torch.no_grad():  # no gradient
            self._update_momentum_encoder_and_decoder(m)  # update the momentum encoder

            # compute momentum features as targets
            k1 = self.decoder(*momentum_features_x)
            k2 = self.decoder(*momentum_features_y)
            # print(decoder_output.shape,decoder_output_y.shape,k1.shape,k2.shape)  #torch.Size([4, 3, 512, 512]) torch.Size([4, 3, 512, 512]) torch.Size([4, 16, 512, 512]) torch.Size([4, 16, 512, 512])
        loss_cl = self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)
        return masks, loss_cl

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x

    @torch.no_grad()
    def _update_momentum_encoder_and_decoder(self, m):
        """Momentum update of the momentum encoder and decoder"""
        for param_b, param_m in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

        for param_b, param_m in zip(self.decoder.parameters(), self.momentum_decoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def contrastive_loss(self, q, k):

        self.T = 0.2
        # print(q.shape,k.shape)  #torch.Size([2, 3, 512, 512]) torch.Size([2, 16, 512, 512])
        # normalize
        q = torch.nn.functional.normalize(q, dim=1)
        k = torch.nn.functional.normalize(k, dim=1)
        print(q.shape,k.shape)
        # gather all targets
        # k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('bcij,bmij->bcij', [q, k]) / self.T
        print(logits)
        N = logits.shape[0]  # batch size per GPU
        labels = torch.arange(N, dtype=torch.long) 
                # + N * torch.distributed.get_rank()).cuda()    #world_size = 0 -> rank = 0
        return torch.nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
