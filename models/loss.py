import torch
import torch.nn as nn
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=1.0):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, im, s):
        im = im / torch.norm(im, dim=1, keepdim=True)
        s = s / torch.norm(s, dim=1, keepdim=True)

        similarity = torch.mm(im, s.transpose(1, 0).contiguous())
        positives = torch.diag(similarity)

        numerator = torch.exp(positives / self.temperature)
        denominator = torch.exp(similarity / self.temperature)

        all_losses = - torch.log(numerator / (torch.sum(denominator, dim=0))) - torch.log(
            numerator / (torch.sum(denominator, dim=1)))
        loss = torch.mean(all_losses)

        return loss