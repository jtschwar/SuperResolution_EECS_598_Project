from torch import nn

class SRCNN(nn.Module):
    def __init__(self, num_channels=3, filters=[64,32], kernels=[9,5]):
        super(SRCNN, self).__init__()
        # patch extraction
        self.conv1 = nn.Conv2D(num_channels,filters[0],kernel_size=kernels[0],padding=kernels[0]//2)
        # non-linear mapping
        self.conv2 = nn.Conv2D(filters[0],filters[1],kernel_size=kernels[1],padding=kernels[1]//2)
        # reconstruction
        self.conv3 = nn.Conv2D(filters[1], num_channels,kernel_size=kernels[1],padding=kernels[1]//2)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class FSRCNN(nn.Module):
    def __init__(self, scale_factor, num_channels, d=56,s=12,m=4):
        super(FSRCNN, self).__init__()
        
        # feature extraction layer
        self.feature_extraction = nn.Sequential(nn.Conv2d(num_channels, d, kernel_size=5,padding=5//2),
                                                nn.PReLU(d))
    
        # Shrinking Layer
        self.shrink = nn.Sequential(nn.Conv2d(d,s,kernel_size=1), nn.PReLU(s))

        # Mapping Layer
        self.map = nn.Sequential([nn.Conv2d(s,s,kernel_size=3,padding=3//2),nn.PReLU(s)] for _ in range(m))

        # Expanding Layer
        self.expand = nn.Sequential(nn.Conv2d(s,d,kernel_size=1), nn.PReLU(d))

        # Deconvolution Layer
        self.deconv = nn.ConvTranspose2d(d, num_channels, kernel_size=9,stride=scale_factor,
                                         padding=9//2, output_padding=scale_factor-1)

        # Initialize model weights
        # tbd...

    def forward(self,x):
        out = self.feature_extraction(x)
        out = self.shrink(x)
        out = self.map(x)
        out = self.expand(x)
        out = self.deconv(x)
    
# Implement VDSR and (SRResNet or EDSR)


# TDB...
# class VDSR(nn.Module):

# TDB...
# class SRResNet(nn.Module):

# TBD...
# class EDSR(nn.Module):