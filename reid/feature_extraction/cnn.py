from __future__ import absolute_import
from collections import OrderedDict
import torch

from ..utils import to_torch

def extract_cnn_feature(model, inputs,training_phase=None):
    model.eval()
    with torch.no_grad():
        inputs = to_torch(inputs).cuda()


        Expand=False
        if inputs.size(0)<2:
            Pad=inputs[:1]
            inputs=torch.cat((inputs,Pad),dim=0)
            Expand=True


        outputs = model(inputs,training_phase=training_phase)
        outputs = outputs.data.cpu()

        if Expand:
            outputs=outputs[:-1]

        return outputs

