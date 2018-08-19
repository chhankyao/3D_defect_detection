from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from my_model import _netlocalD, _netG, _netG_resNet, _netlocalD_patch, UnetGenerator
import utils
import numpy as np
import numpy.ma as ma
import datetime
import time
import sys
import shutil
# log
class Logger(object):
	def __init__(self, f1, f2):
		self.f1, self.f2 = f1, f2

	def write(self, msg):
		self.f1.write(msg)
		self.f2.write(msg)
		
	def flush(self):
		pass

# Clear directory
def clear_dir(folder):
	for the_file in os.listdir(folder):
		file_path = os.path.join(folder, the_file)
		try:
			if os.path.isfile(file_path):
				os.unlink(file_path)
			elif os.path.isdir(file_path):
				shutil.rmtree(file_path)
		except Exception as e:
			print(e)
	return 


# data format: context_image_path, entire_image_path, mask_path
def default_flist_reader(flist):
	imlist = []
	with open(flist, 'r') as rf:
		for line in rf.readlines():
			context_image_path, entire_image_path, mask_path = line.strip().split(' ')
			imlist.append( (context_image_path, entire_image_path, mask_path) )
					
	return imlist

# do not use io.imread(path), otherwise there will be error "'int' object is not subscriptable"
def default_image_loader(path):
	with open(path, 'rb') as f:
		with Image.open(f) as image:
			return image.convert('RGB')	


# data format: context_image, entire_image, mask
class ImageContextDataset(Dataset):
	def __init__(self, flist, flist_reader=default_flist_reader, loader=default_image_loader, 
		transform=None, mask_transform=None):	
		self.transform = transform
		self.loader = loader
		self.imlist = flist_reader(flist)
		self.mask_transform = mask_transform

		print("Read "+str(len(self.imlist))+" data samples.")


	def __getitem__(self, index):
		context_image_path, entire_image_path, mask_path = self.imlist[index]
		context_image = self.loader(context_image_path)
		entire_image = self.loader(entire_image_path)
		mask = np.load(mask_path)
		mask = mask.astype('uint8')*255  # for visualize the image or save the image
		
		mask_3d = np.dstack((mask,mask,mask))
		
		mask_image = Image.fromarray(mask_3d, 'RGB')


		if self.transform is not None:
			context_image = self.transform(context_image)
			entire_image = self.transform(entire_image)	

		if self.mask_transform is not None:	
			mask_image = self.mask_transform(mask_image)

		
		return context_image, entire_image, mask_image



	def __len__(self):
		return len(self.imlist)

parser = argparse.ArgumentParser()
parser.add_argument('--mask',  default='segment', help='box | segment')
parser.add_argument('--ride',  default='horse', help='horse | bicycle')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')

parser.add_argument('--resume_epoch', type=int, default=0, help='epoch id from which training is resumed')


parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64, help='#  of gen filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='#  of discrim filters in first conv layer')
parser.add_argument('--nc', type=int, default=3, help='# of channels in input')

parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')

parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

parser.add_argument('--nBottleneck', type=int,default=4000,help='of dim for bottleneck of encoder')
parser.add_argument('--overlapPred',type=int,default=4,help='overlapping edges')
parser.add_argument('--nef',type=int,default=64,help='of encoder filters in first conv layer')
parser.add_argument('--wtl2',type=float,default=0.75,help='0 means do not use else use with this weight')
parser.add_argument('--arch',  default='res', help='context | res-patch | ushape')

opt = parser.parse_args()
print(opt)

# make dir
exp_name = opt.mask+"_phone"
# if not exist, make dirs
result_train_dir = os.path.join("train_"+str(opt.arch), exp_name)
if not os.path.exists(result_train_dir):
	os.makedirs(result_train_dir)


train_cropped_dir = os.path.join(result_train_dir, 'cropped')
if not os.path.exists(train_cropped_dir):
	os.makedirs(train_cropped_dir)
train_real_dir = os.path.join(result_train_dir, 'real')
if not os.path.exists(train_real_dir):
	os.makedirs(train_real_dir)
train_predict_dir = os.path.join(result_train_dir, 'predict')
if not os.path.exists(train_predict_dir):
	os.makedirs(train_predict_dir)
train_model_dir = os.path.join(result_train_dir, 'model')
if not os.path.exists(train_model_dir):
	os.makedirs(train_model_dir)
train_log_dir = os.path.join(result_train_dir, 'log')	
if not os.path.exists(train_log_dir):
	os.makedirs(train_log_dir)
train_loss_dir = os.path.join(result_train_dir, 'loss')	
if not os.path.exists(train_loss_dir):
	os.makedirs(train_loss_dir)

# redirect
current_time = time.time()
st = datetime.datetime.fromtimestamp(current_time).strftime('%Y-%m-%d_%H:%M:%S')
logfile = open(os.path.join(train_log_dir, 'train_'+st+'.txt'), 'w')
sys.stdout = Logger(sys.stdout, logfile)	

# set seed
if opt.manualSeed is None:
	opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
	torch.cuda.manual_seed_all(opt.manualSeed)

# set cuda
cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
	print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# set datasets
transform = transforms.Compose([transforms.Scale((opt.imageSize,opt.imageSize)),
								#transforms.CenterCrop(opt.imageSize),
								transforms.ToTensor(),
								transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# ToTensor: Converts a PIL Image or numpy.ndarray (H x W x C) in the range
# [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
mask_transform = transforms.Compose([transforms.Scale((opt.imageSize,opt.imageSize)),
								transforms.ToTensor()])

triplet_dir = ""
train_file = "projection3_train.txt"
# train_file = opt.mask+"_train_man_ride_"+opt.ride+"_mask_person.txt"

dataset = ImageContextDataset(os.path.join(triplet_dir, train_file), flist_reader=default_flist_reader, loader=default_image_loader, 
		transform=transform, mask_transform=mask_transform)   
assert dataset

# data loader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
										 shuffle=True, num_workers=int(opt.workers))
# parameters
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = int(opt.nc)  # number of channels
nef = int(opt.nef)
nBottleneck = int(opt.nBottleneck)
wtl2 = float(opt.wtl2)
print("==> wtl2=", wtl2)
overlapL2Weight = 10

# custom weights initialization called on netG and netD
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)

# start epoch
resume_epoch = opt.resume_epoch

# initialize or resume generative model
if opt.arch == "context":
	netG = _netG(opt)
	print('==> Generative model: context')
elif opt.arch == "res-patch":
	netG = _netG_resNet()
	print('==> Generative model: resNet')
elif opt.arch == "ushape":
	netG = UnetGenerator(3, 3, 7, opt.ngf)	

netG.apply(weights_init)
if opt.netG != '':
	netG.load_state_dict(torch.load(opt.netG,map_location=lambda storage, location: storage)['state_dict'])
	resume_epoch = torch.load(opt.netG)['epoch']

print(netG)

# initialize or resume discriminative model
if opt.imageSize == 128:
	netD = _netlocalD(opt)
	print('==> Discriminative model: patch 128*128')
elif opt.imageSize == 256:
	netD = _netlocalD_patch()
	print('==> Discriminative model: patch 256*256')	


netD.apply(weights_init)
if opt.netD != '':
	netD.load_state_dict(torch.load(opt.netD,map_location=lambda storage, location: storage)['state_dict'])
	resume_epoch = torch.load(opt.netD)['epoch']

print(netD)

print('==> Start from epoch '+str(resume_epoch))
if resume_epoch == 0:
	clear_dir(train_cropped_dir)
	clear_dir(train_real_dir)
	clear_dir(train_predict_dir)
	clear_dir(train_model_dir)
	clear_dir(train_log_dir)
	clear_dir(train_loss_dir)

# loss functions
criterion = nn.BCELoss()

# label values
real_label = 1
fake_label = 0

# initialize the variables
label = torch.FloatTensor(opt.batchSize,1)
label = torch.autograd.Variable(label)


if opt.cuda:
	netD.cuda()
	netG.cuda()
	criterion.cuda()
	label = label.cuda()
	

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

if resume_epoch == 0:
	Loss_D_log = []
	Loss_G_log = []
	errG_D_log = []
	errG_l2_log = []
else:
	Loss_D_log = np.load('{0}/Loss_D.npy'.format(train_loss_dir))
	Loss_G_log = np.load('{0}/Loss_G.npy'.format(train_loss_dir))
	errG_D_log = np.load('{0}/errG_D.npy'.format(train_loss_dir))
	errG_l2_log = np.load('{0}/errG_l2.npy'.format(train_loss_dir))
		
# training
for epoch in list(range(resume_epoch, opt.niter)):
	Loss_D_cur = 0.0
	Loss_G_cur = 0.0
	errG_D_cur = 0.0
	errG_l2_cur = 0.0

	for i, (context_image, entire_image, mask) in enumerate(dataloader):
		
		context_image_var = torch.autograd.Variable(context_image)
		entire_image_var = torch.autograd.Variable(entire_image)
		mask_var = torch.autograd.Variable(mask)
		
		if opt.cuda:
			context_image_var = context_image_var.cuda()
			entire_image_var = entire_image_var.cuda()
			mask_var = mask_var.cuda()
	
		
		############################
		# (1) Update D network
		###########################

		# train with real images
		netD.zero_grad()
		cur_batch_size = context_image_var.size()[0]
		label.data.resize_(cur_batch_size,1).fill_(real_label)

		output = netD(entire_image_var)
		errD_real = criterion(output, label)
		errD_real.backward()

		# train with fake images
		fake_image = netG(context_image_var)
		label.data.fill_(fake_label)
		output = netD(fake_image.detach())
		errD_fake = criterion(output, label)
		errD_fake.backward()
	
		# total loss for discriminator
		errD = errD_real + errD_fake
		Loss_D_cur += errD.data[0]

		optimizerD.step()


		############################
		# (2) Update G network: maximize log(D(G(z)))
		###########################
		netG.zero_grad()
		label.data.fill_(real_label) 
		output = netD(fake_image)
		
                # adversarial loss
		errG_D = criterion(output, label)
		
                # l2 loss
		errG_l2 = (fake_image-entire_image_var).pow(2)
		errG_l2 = errG_l2 * mask_var
		errG_l2 = errG_l2.mean()
		
                # total loss for generator
		errG = (1-wtl2) * errG_D + wtl2 * errG_l2

		Loss_G_cur += errG.data[0]
		errG_D_cur += errG_D.data[0]
		errG_l2_cur += errG_l2.data[0]

		errG.backward()

		D_G_z2 = output.data.mean()
		optimizerG.step()

		print('[Epoch: %d/%d][Iter: %d/%d] Loss_D: %.4f Loss_G: %.4f [errG_D: %.4f / errG_l2: %.4f]'
			  % (epoch+1, opt.niter, i+1, len(dataloader),
				 errD.data[0], errG.data[0], errG_D.data[0], errG_l2.data[0]))

		# save resulting image every 100 iterations
		if i % 100 == 0:
			fake_image_numpy = fake_image.data.cpu().numpy()*0.5+0.5
			context_image_numpy = context_image_var.data.cpu().numpy()*0.5+0.5
			entire_image_numpy = entire_image_var.data.cpu().numpy()*0.5+0.5
			vutils.save_image(torch.from_numpy(entire_image_numpy),
					os.path.join(train_real_dir, 'real_epoch_%03d_iter_%03d.png' % (epoch, i)) )
			vutils.save_image(torch.from_numpy(context_image_numpy),
					os.path.join(train_cropped_dir, 'cropped_epoch_%03d_iter_%03d.png' % (epoch, i)) )
			vutils.save_image(torch.from_numpy(fake_image_numpy),
					os.path.join(train_predict_dir, 'predict_epoch_%03d_iter_%03d.png' % (epoch, i)) )		

	# update loss
	Loss_D_cur /= float(len(dataloader))
	Loss_G_cur /= float(len(dataloader))
	errG_D_cur /= float(len(dataloader))
	errG_l2_cur /= float(len(dataloader))

	Loss_D_log.append(Loss_D_cur)
	Loss_G_log.append(Loss_G_cur)
	errG_D_log.append(errG_D_cur)
	errG_l2_log.append(errG_l2_cur)

	# save loss
	np.save('{0}/Loss_D.npy'.format(train_loss_dir), np.array(Loss_D_log))
	np.save('{0}/Loss_G.npy'.format(train_loss_dir), np.array(Loss_G_log))
	np.save('{0}/errG_D.npy'.format(train_loss_dir), np.array(errG_D_log))
	np.save('{0}/errG_l2.npy'.format(train_loss_dir), np.array(errG_l2_log))

	# save checkpoints (must save the last epoch)
	if epoch % 5 == 0 or epoch == opt.niter - 1:
		torch.save({'epoch':epoch,
					'state_dict':netG.state_dict()},
					os.path.join(train_model_dir,'netG_epoch_%03d.pth' % (epoch)) )
		torch.save({'epoch':epoch,
					'state_dict':netD.state_dict()},
					os.path.join(train_model_dir, 'netD_epoch_%03d.pth' % (epoch)) )
				
