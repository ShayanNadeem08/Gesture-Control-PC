
import torch
from torch import nn

# Basic 3D convolution block
class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv3DBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# Dense Block for 3D DenseNet
class DenseBlock3D(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock3D, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(self._make_dense_layer(in_channels + i * growth_rate, growth_rate))

    def _make_dense_layer(self, in_channels, growth_rate):
        return nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, 4 * growth_rate, kernel_size=1, bias=False),
            nn.BatchNorm3d(4 * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv3d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, 1))
            features.append(new_feature)
        return torch.cat(features, 1)


# Transition Layer for 3D DenseNet
class TransitionLayer3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer3D, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool3d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.layers(x)


# 3D DenseNet Model
class DenseNet3D(nn.Module):
    def __init__(self, growth_rate=12, block_config=(2, 4, 6), num__init__features=16,
                 compression_rate=0.5, num_classes=4):
        super(DenseNet3D, self).__init__()

        # First convolution and pooling
        self.features = nn.Sequential(
            nn.Conv3d(1, num__init__features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(num__init__features),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        # Dense Blocks
        num_features = num__init__features
        for i, num_layers in enumerate(block_config):
            # Add a dense block
            block = DenseBlock3D(
                in_channels=num_features,
                growth_rate=growth_rate,
                num_layers=num_layers
            )
            self.features.add_module(f'denseblock{i+1}', block)
            num_features += num_layers * growth_rate

            # Add a transition layer (except after the last block)
            if i != len(block_config) - 1:
                out_features = int(num_features * compression_rate)
                trans = TransitionLayer3D(num_features, out_features)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = out_features

        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm3d(num_features))
        self.features.add_module('relu_final', nn.ReLU(inplace=True))

        # Global Average Pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Linear(num_features, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = self.avgpool(features)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out