import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

class ResBlock(nn.Module):
	def __init__(self, n, size, isInstanceNorm=True):
		super(ResBlock, self).__init__()
		if isInstanceNorm == True:
			normLayer = nn.InstanceNorm2d
		else:
			normLayer = nn.BatchNorm2d
		self.norm1   = normLayer(n)
		self.norm2   = normLayer(n)
		self.conv1 = nn.Conv2d(n, n, size, padding=1, bias=False)
		self.conv2 = nn.Conv2d(n, n, size, padding=1, bias=True)

	def forward(self, x):
		y = x
		y = F.relu(self.norm1(self.conv1(y)), True)
		y = self.norm2(self.conv2(y))
		return x+y

class _netG_resNet(nn.Module):
	def __init__(self, isInstanceNorm=True):
		super(_netG_resNet, self).__init__()

		if isInstanceNorm:
			normLayer = nn.InstanceNorm2d
		else:
			normLayer = nn.BatchNorm2d

		self.conv1 = nn.Conv2d(in_channels=3, out_channels=32,
				kernel_size=7, stride=1, padding=3, bias=False)
		self.norm1 = normLayer(32)
		self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
				kernel_size=3, stride=2, padding=1, bias=False)
		self.norm2 = normLayer(64)
		self.conv3 = nn.Conv2d(in_channels=64, out_channels=128,
				kernel_size=3, stride=2, padding=1, bias=False)
		self.norm3 = normLayer(128)

		self.res1 = ResBlock(128, 3, isInstanceNorm)
		self.res2 = ResBlock(128, 3, isInstanceNorm)
		self.res3 = ResBlock(128, 3, isInstanceNorm)
		self.res4 = ResBlock(128, 3, isInstanceNorm)
		self.res5 = ResBlock(128, 3, isInstanceNorm)
		self.res6 = ResBlock(128, 3, isInstanceNorm)
		self.res7 = ResBlock(128, 3, isInstanceNorm)
		self.res8 = ResBlock(128, 3, isInstanceNorm)
		self.res9 = ResBlock(128, 3, isInstanceNorm)

		self.dconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64,
				kernel_size=4, stride=2, padding=1, output_padding=0, bias=False)
		self.dnorm1 = normLayer(64)
		self.dconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32,
				kernel_size=4, stride=2, padding=1, output_padding=0, bias=False)
		self.dnorm2 = normLayer(32)
		self.conv4 = nn.Conv2d(in_channels=32, out_channels=3,
				kernel_size=7, stride=1, padding=3, bias=True)

	def forward(self, x):
		x = F.relu(self.norm1(self.conv1(x)), True)
		x = F.relu(self.norm2(self.conv2(x)), True)
		x = F.relu(self.norm3(self.conv3(x)), True)

		x = self.res1(x)
		x = self.res2(x)
		x = self.res3(x)
		x = self.res4(x)
		x = self.res5(x)
		x = self.res6(x)
		x = self.res7(x)
		x = self.res8(x)
		x = self.res9(x)

		x = F.relu(self.dnorm1(self.dconv1(x)), True)
		x = F.relu(self.dnorm2(self.dconv2(x)), True)
		x = F.tanh(self.conv4(x))
		return x


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
	def __init__(self, input_nc, output_nc, num_downs, ngf=64,
				 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
		super(UnetGenerator, self).__init__()
		self.gpu_ids = gpu_ids

		# construct unet structure
		unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
		for i in range(num_downs - 5):
			unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
		unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
		unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
		unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
		unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

		self.model = unet_block

	def forward(self, input):
		if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
			return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
		else:
			return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
	def __init__(self, outer_nc, inner_nc, input_nc=None,
				 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
		super(UnetSkipConnectionBlock, self).__init__()
		self.outermost = outermost
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d
		if input_nc is None:
			input_nc = outer_nc
		downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
							 stride=2, padding=1, bias=use_bias)
		downrelu = nn.LeakyReLU(0.2, True)
		downnorm = norm_layer(inner_nc)
		uprelu = nn.ReLU(True)
		upnorm = norm_layer(outer_nc)

		if outermost:
			upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
										kernel_size=4, stride=2,
										padding=1)
			down = [downconv]
			up = [uprelu, upconv, nn.Tanh()]
			model = down + [submodule] + up
		elif innermost:
			upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
										kernel_size=4, stride=2,
										padding=1, bias=use_bias)
			down = [downrelu, downconv]
			up = [uprelu, upconv, upnorm]
			model = down + up
		else:
			upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
										kernel_size=4, stride=2,
										padding=1, bias=use_bias)
			down = [downrelu, downconv, downnorm]
			up = [uprelu, upconv, upnorm]

			if use_dropout:
				model = down + [submodule] + up + [nn.Dropout(0.5)]
			else:
				model = down + [submodule] + up

		self.model = nn.Sequential(*model)

	def forward(self, x):
		if self.outermost:
			return self.model(x)
		else:
			return torch.cat([x, self.model(x)], 1)

class _netG(nn.Module):
	def __init__(self, opt):
		super(_netG, self).__init__()
		self.ngpu = opt.ngpu
		self.main = nn.Sequential(
			# input is (nc) x 128 x 128
			nn.Conv2d(opt.nc,opt.nef,4,2,1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size: (nef) x 64 x 64
			nn.Conv2d(opt.nef,opt.nef,4,2,1, bias=False),
			nn.BatchNorm2d(opt.nef),
			nn.LeakyReLU(0.2, inplace=True),
			# state size: (nef) x 32 x 32
			nn.Conv2d(opt.nef,opt.nef*2,4,2,1, bias=False),
			nn.BatchNorm2d(opt.nef*2),
			nn.LeakyReLU(0.2, inplace=True),
			# state size: (nef*2) x 16 x 16
			nn.Conv2d(opt.nef*2,opt.nef*4,4,2,1, bias=False),
			nn.BatchNorm2d(opt.nef*4),
			nn.LeakyReLU(0.2, inplace=True),
			# state size: (nef*4) x 8 x 8
			nn.Conv2d(opt.nef*4,opt.nef*8,4,2,1, bias=False),
			nn.BatchNorm2d(opt.nef*8),
			nn.LeakyReLU(0.2, inplace=True),
			# state size: (nef*8) x 4 x 4
			nn.Conv2d(opt.nef*8,opt.nBottleneck,4, bias=False),
			# tate size: (nBottleneck) x 1 x 1
			nn.BatchNorm2d(opt.nBottleneck),
			nn.LeakyReLU(0.2, inplace=True),
			# input is Bottleneck, going into a convolution
			nn.ConvTranspose2d(opt.nBottleneck, opt.ngf * 8, 4, 1, 0, bias=False),
			nn.BatchNorm2d(opt.ngf * 8),
			nn.ReLU(True),
			# state size. (ngf*8) x 4 x 4
			nn.ConvTranspose2d(opt.ngf * 8, opt.ngf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(opt.ngf * 4),
			nn.ReLU(True),
			# state size. (ngf*4) x 8 x 8
			nn.ConvTranspose2d(opt.ngf * 4, opt.ngf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(opt.ngf * 2),
			nn.ReLU(True),
			# state size. (ngf*2) x 16 x 16
			nn.ConvTranspose2d(opt.ngf * 2, opt.ngf, 4, 2, 1, bias=False),
			nn.BatchNorm2d(opt.ngf),
			nn.ReLU(True),
			# state size. (ngf) x 32 x 32
			nn.ConvTranspose2d(opt.ngf, opt.ngf, 4, 2, 1, bias=False),
			nn.BatchNorm2d(opt.ngf),
			nn.ReLU(True),
			# state size. (ngf) x 64 x 64
			nn.ConvTranspose2d(opt.ngf, opt.nc, 4, 2, 1, bias=False),
			nn.Tanh()
			# state size. (nc) x 128 x 128
		)

	def forward(self, input):
		if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
			output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
		else:
			output = self.main(input)
		return output

# input: 128*128
class _netlocalD(nn.Module):
	def __init__(self, opt):
		super(_netlocalD, self).__init__()
		self.ngpu = opt.ngpu
		self.main = nn.Sequential(
			# input is (nc) x 128 x 128
			nn.Conv2d(opt.nc, opt.ndf, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# input is (nc) x 64 x 64
			nn.Conv2d(opt.ndf, opt.ndf, 4, 2, 1, bias=False),
			nn.BatchNorm2d(opt.ndf),
			nn.LeakyReLU(0.2, inplace=True),
			# input is (ndf) x 32 x 32
			nn.Conv2d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(opt.ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# input is (ndf*2) x 16 x 16
			nn.Conv2d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(opt.ndf * 4),
			nn.LeakyReLU(0.2, inplace=True),
			# input is (ndf*4) x 8 x 8
			nn.Conv2d(opt.ndf * 4, opt.ndf * 8, 4, 2, 1, bias=False),
			nn.BatchNorm2d(opt.ndf * 8),
			nn.LeakyReLU(0.2, inplace=True),
			# input is (ndf*8) x 4 x 4
			nn.Conv2d(opt.ndf * 8, 1, 4, 1, 0, bias=False),
			nn.Sigmoid()
		)

	def forward(self, input):
		if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
			output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
		else:
			output = self.main(input)

		return output.view(-1, 1)

# input: 256*256
class _netlocalD_patch(nn.Module):
	def __init__(self):
		super(_netlocalD_patch, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, 4, 2, 1, bias=False)
		self.lre1 = nn.LeakyReLU(0.2, inplace=True)

		self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
		self.bn2 = nn.BatchNorm2d(128)
		self.lre2 = nn.LeakyReLU(0.2, inplace=True)

		self.conv3 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
		self.bn3 = nn.BatchNorm2d(256)
		self.lre3 = nn.LeakyReLU(0.2, inplace=True)

		self.conv4 = nn.Conv2d(256, 512, 4, 2, 1, bias=False)
		self.bn4 = nn.BatchNorm2d(512)
		self.lre4 = nn.LeakyReLU(0.2, inplace=True)

		self.conv5 = nn.Conv2d(512, 1, 1, 1, 0, bias=False)
		self.sig = nn.Sigmoid()

	def forward(self, x):
		x = self.lre1(self.conv1(x))
		x = self.lre2(self.bn2(self.conv2(x)))
		x = self.lre3(self.bn3(self.conv3(x)))
		x = self.lre4(self.bn4(self.conv4(x)))
		x = self.sig(self.conv5(x))
		return x
