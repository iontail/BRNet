from torch import nn
from typing import Dict, List, Tuple, Optional, Any, Union
from transformers.trainer import Trainer
from torch import nn
import torch
import numpy as np

class BRTrainer(Trainer):
    def __init__(self, net_enh=None, **kwds):
        super().__init__(**kwds)
        self.net_enh = net_enh
        self.loss_fn = nn.CrossEntropyLoss()
        
    def compute_loss(self, model, inputs, num_items_in_batch = None, return_outputs=False):
        image = inputs['image']
        label = inputs['label']
        darklevel = inputs['darklevel']
        img_path = inputs['img_path']
        
        images = image.to(self.device) / 255.0
        detection_targets = [cls.to(self.device) for cls in label]
        # Forward pass
        output = model(image)
        # Compute loss
        loss = self.loss_fn(output, label)

        if return_outputs:
            return loss, output, label
        return loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        model.eval()
        
        with torch.no_grad():
            eval_loss, pred, label = self.compute_loss(model,inputs,return_outputs = True)
        
        return (eval_loss,pred,label)