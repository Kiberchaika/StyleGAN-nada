import torch
from torch import nn
from models.encoders.model_irse import Backbone


class IDLoss(nn.Module):
    def __init__(self, opts):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(opts.arcface_model_paths))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.face_prepare = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.facenet.eval()
        self.opts = opts

    def extract_feats(self, x):
        x = self.face_prepare(x)
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, x, y):
        n_samples = x.shape[0]
        x_feats = self.extract_feats(x)
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_feats = y_feats.detach()

        loss = 0
        count = 0

        for i in range(n_samples):
            # torch.nn.CosineSimilarity([x_feats, y_feats])
            #loss += torch.pow(x_feats[i].dot(y_feats[i]),2)
            loss += torch.pow(x_feats[i] - y_feats[i], 2).mean()
            count += 1

        return loss / count
