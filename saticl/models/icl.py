from typing import List, Tuple, Type

import torch
from torch import nn

from saticl.models.base import Decoder, Encoder, Head


class ClassifierHead(nn.Module):

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        # this contains the number of classes in the current step
        # e.g. with steps 0 1 | 4 5 | 3 6 7, num classes will be 2 | 2 | 3
        self.num_classes = num_classes
        self.out = nn.Conv2d(in_channels=in_channels, out_channels=num_classes, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(x)

    def init_weights(self, old_classifier: "ClassifierHead") -> None:
        """Initializes the current head from the old classifier instance,
        with pretrained weights already loaded.

        Args:
            old_classifier (ClassifierHead): old instance with pretrained weights on classes C^{t-1}
        """
        # weights: [num classes, in_channels, ksize, ksize], bias: [num_classes]
        # get old weights and bias from the class 0, background
        old_weights = old_classifier.out.weight[0]
        old_bias = old_classifier.out.bias[0]
        # init bias with oldB - log(|Ct|)
        bias_diff = torch.tensor([self.num_classes + 1], device=old_weights.device, dtype=torch.float)
        new_bias = old_bias - torch.log(bias_diff)
        self.out.weight.data.copy_(old_weights)
        self.out.bias.data.copy_(new_bias)


class ICLSegmenter(nn.Module):

    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 classes: List[int],
                 head: Type[Head] = ClassifierHead,
                 return_features: bool = False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        assert isinstance(classes, (list, tuple)), "Classes expected to be a list of class counts per step"
        classifiers = [head(in_channels=decoder.output(), num_classes=count) for count in classes]
        self.classifiers = nn.ModuleList(classifiers)
        self.classes = classes
        self.classes_total = sum(classes)
        self.return_features = return_features

    def forward_features(self, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        encoder_out = self.encoder(inputs)
        decoder_out = self.decoder(encoder_out)
        head_out = torch.cat([head(decoder_out) for head in self.classifiers], dim=1)
        return head_out, encoder_out

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        out, features = self.forward_features(inputs)
        features = features if self.return_features else None
        return out, features

    def init_classifier(self):
        assert len(self.classifiers) >= 2, "Cannot init a new classifier at step 0!"
        self.classifiers[-1].init_weights(self.classifiers[-2])

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
