import torch.nn as nn
import torchvision.models as models
import timm, torch
import numpy as np

class efficientformer_color(nn.Module):  
    def __init__(self):
        super(efficientformer_color, self).__init__()

        self.encoder = timm.create_model('efficientformerv2_s1', pretrained=False, num_classes=0, in_chans=15)
        self.color_linear = nn.Linear(224, 112)
        self.batch_norm = nn.BatchNorm1d(112)
        self.relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.5)

        self.color_linear2 = nn.Linear(112, 56)
        self.batch_norm2 = nn.BatchNorm1d(56)
        self.relu2 = nn.LeakyReLU(0.2)
        self.dropout2 = nn.Dropout(0.3)
        
        self.color_linear3 = nn.Linear(56, 18)
        
    def forward(self, x):
        """ Forward propagation with input 'x' """
        feat_rgb = x['image_rgb'].float()
        feat_hsv = x['image_hsv'].float()
        feat_lab = x['image_lab'].float()
        feat_hls = x['image_hls'].float()
        feat_yuv = x['image_yuv'].float()
        
        feat = torch.cat((feat_hsv, feat_lab, feat_rgb, feat_hls, feat_yuv), 1)
        feat_out = self.encoder(feat)

        out = self.color_linear(feat_out)
        out = self.batch_norm(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.color_linear2(out)
        out = self.batch_norm2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        out = self.color_linear3(out)
        
        return out

if __name__ == '__main__':
    pass
