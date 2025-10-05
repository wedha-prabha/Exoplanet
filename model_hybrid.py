# src/model_hybrid.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    def __init__(self, in_ch=1, channels=[16,32,64], kernel=5):
        super().__init__()
        layers=[]
        cur_ch = in_ch
        for c in channels:
            layers.append(nn.Conv1d(cur_ch, c, kernel_size=kernel, padding=kernel//2, stride=2))
            layers.append(nn.BatchNorm1d(c))
            layers.append(nn.ReLU())
            cur_ch = c
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: B x 1 x L
        return self.net(x)   # B x C x Ldown

class SimpleTransformerEncoder(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   activation="relu")
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # positional embedding
        self.pos_emb = None

    def forward(self, x):
        # x: B x L x D (transformer expects seq_len x batch x d_model)
        x = x.permute(1,0,2)
        out = self.transformer(x)
        out = out.permute(1,0,2)
        return out

class HybridModel(nn.Module):
    def __init__(self, input_len=512, cnn_channels=[16,32,64], transformer_dim=64, num_classes=1):
        super().__init__()
        self.cnn = CNNEncoder(in_ch=1, channels=cnn_channels)
        # compute downsampled length after cnn strides: each conv has stride 2, so downsample factor = 2^len(channels)
        down_factor = 2 ** len(cnn_channels)
        self.down_len = input_len // down_factor
        self.proj = nn.Conv1d(cnn_channels[-1], transformer_dim, kernel_size=1)
        self.transformer = SimpleTransformerEncoder(d_model=transformer_dim, nhead=4, num_layers=2)
        # pooling and heads
        self.classifier = nn.Sequential(
            nn.Linear(transformer_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        # regression head for (period, depth, duration)
        self.regressor = nn.Sequential(
            nn.Linear(transformer_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        # x: B x 1 x L
        feat = self.cnn(x)  # B x C x L'
        feat = self.proj(feat)  # B x D x L'
        feat = feat.permute(0,2,1)  # B x L' x D
        tr = self.transformer(feat) # B x L' x D
        # global pooling (mean)
        pooled = tr.mean(dim=1)
        cls_logits = self.classifier(pooled)
        reg = self.regressor(pooled)
        return cls_logits.squeeze(-1), reg

if __name__ == "__main__":
    model = HybridModel(input_len=512)
    import torch
    x = torch.randn(4,1,512)
    cls, reg = model(x)
    print(cls.shape, reg.shape)
