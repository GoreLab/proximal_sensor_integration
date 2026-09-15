import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransposedResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(TransposedResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride==1:
            self.upsample = nn.Identity()
        else:
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(indim, outdim, kernel_size=3, padding=1, output_padding=1, stride=stride),
                nn.BatchNorm2d(outdim)
            )
            
 
        self.block = nn.Sequential(
            nn.ConvTranspose2d(indim, outdim, kernel_size=3, padding=1, output_padding=1, stride=stride), # spatial + channel change
            nn.BatchNorm2d(outdim),
            nn.ReLU(inplace=True),
            nn.Conv2d(outdim, outdim, kernel_size=3, padding=1), # channel change
            nn.BatchNorm2d(outdim)
        )

 
    def forward(self, x):
        return F.relu(self.block(x) + self.upsample(x))


class Decoder(nn.Module):
    def __init__(self, in_feats, out_channels, n_blocks):
        super(Decoder, self).__init__()

        step = (in_feats - out_channels) // n_blocks

        hidden_dims = [in_feats]
        for i in range(n_blocks - 1):
            hidden_dims.append(in_feats - (i + 1)*step)

        modules = []
        for i in range(n_blocks - 1):
            modules.append(
                TransposedResBlock(hidden_dims[i], hidden_dims[i + 1], stride=2)
            )

        modules.append(TransposedResBlock(hidden_dims[-1], hidden_dims[-1], stride=2))
        modules.append(nn.Conv2d(hidden_dims[-1], out_channels, kernel_size=3, padding=1))
        modules.append(nn.Tanh())

        self.decoder = nn.Sequential(*modules)

    def forward(self, feats):
        return self.decoder(feats)


class ResnetAEVarLS(nn.Module):
    def __init__(self, in_channels, model_name="resnet18", num_latent_dims=512, **kwargs):
        super(ResnetAEVarLS, self).__init__()

        step = int(math.ceil((num_latent_dims - in_channels) / 4))
        new_map = {64: in_channels + step, 128: in_channels + 2*step, 256: in_channels + 3*step, 512: num_latent_dims}

        self.encoder = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)
        self.encoder.conv1 = nn.Conv2d(in_channels, in_channels + step, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        children = list(self.encoder.children())[:-2]
        for i, child in enumerate(children):
            if i == 0:
                continue
            if isinstance(child, nn.Sequential):
                for basic_block in child:
                    basic_block.conv1 = nn.Conv2d(new_map[basic_block.conv1.in_channels], new_map[basic_block.conv1.out_channels], kernel_size=basic_block.conv1.kernel_size,
                                                  stride=basic_block.conv1.stride, padding=basic_block.conv1.padding, bias=basic_block.conv1.bias)
                    basic_block.bn1   = nn.BatchNorm2d(new_map[basic_block.bn1.num_features])
                    basic_block.conv2 = nn.Conv2d(new_map[basic_block.conv2.in_channels], new_map[basic_block.conv2.out_channels], kernel_size=basic_block.conv2.kernel_size,
                                                  stride=basic_block.conv2.stride, padding=basic_block.conv2.padding, bias=basic_block.conv2.bias)
                    basic_block.bn2   = nn.BatchNorm2d(new_map[basic_block.bn2.num_features])
                    if basic_block.downsample:
                        basic_block.downsample[0] = nn.Conv2d(new_map[basic_block.downsample[0].in_channels], new_map[basic_block.downsample[0].out_channels], kernel_size=basic_block.downsample[0].kernel_size,
                                                        stride=basic_block.downsample[0].stride, padding=basic_block.downsample[0].padding, bias=basic_block.downsample[0].bias)
                        basic_block.downsample[1] = nn.BatchNorm2d(new_map[basic_block.downsample[1].num_features])
            else:
                if isinstance(child, nn.Conv2d):
                    children[i] = nn.Conv2d(new_map[child.in_channels], new_map[child.out_channels], kernel_size=child.kernel_size, stride=child.stride, padding=child.padding, bias=child.bias)
                elif isinstance(child, nn.BatchNorm2d):
                    children[i] = nn.BatchNorm2d(new_map[child.num_features])

        self.encoder = nn.Sequential(*children)
        self.decoder = Decoder(num_latent_dims, in_channels, n_blocks=5) # something about how much was downsampled

    def forward(self, input, **kwargs):
        recons = self.decoder(self.encoder(input)) 
        return [recons, input]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons, input = args
        recons_loss = F.mse_loss(recons, input)
        return {'loss': recons_loss}

    def generate(self, x, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self(x)[0]

if __name__ == "__main__":
    model = ResnetAEVarLS(5)
    print(model.decoder)