import torch
from . import initialization as init


class SegmentationModel_1125(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.decoder)
        if self.projection_head is not None:
            init.initialize_head(self.projection_head)

        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)
    
    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)

        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)
        
        if self.classification_head is not None:

            labels = self.classification_head(features[-1])
            return masks, labels

        if self.projection_head is not None:
            mlp_output = self.projection_head(features[-1].view(-1,features[-1].shape[1]))
            return masks,mlp_output
        return masks

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
            x = self.forward(x,None)

        return x