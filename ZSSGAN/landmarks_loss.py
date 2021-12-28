import torch
from torch import nn
from models.facemesh import FaceMesh

class LandmarksLoss(nn.Module):
    def __init__(self, opts):
        super(LandmarksLoss, self).__init__()
        print('Loading Facemesh')

        self.net = FaceMesh()
        self.net.load_weights(opts.facemesh_model_paths)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((192, 192))
        self.net.eval()
        self.opts = opts

    def extract_feats(self, x):
        x = self.face_pool(x)
        detections, conf = self.net.predict_on_batch(x, x.requires_grad)
        return detections

    def forward(self, x, y):
        x_feats = self.extract_feats(x)
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_feats = y_feats.detach()

        x_feats_px = x_feats[:, 0]
        x_feats_py = x_feats[:, 1]

        y_feats_px = y_feats[:, 0]
        y_feats_py = y_feats[:, 1]

        loss = torch.sqrt(torch.pow(x_feats_px - y_feats_px,2) + torch.pow(x_feats_py - y_feats_py,2)) / 192
        loss = loss.mean()
        return loss
