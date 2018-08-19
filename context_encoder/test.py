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
	def __init__(self, flist, flist_reader=default_flist_reader, loader=default_image_loader, transform=None):	
		self.transform = transform
		self.loader = loader
		self.imlist = flist_reader(flist)

		print("Read "+str(len(self.imlist))+" data samples.")

	def __getitem__(self, index):
		context_image_path, entire_image_path, _ = self.imlist[index]
		context_image = self.loader(context_image_path)
		entire_image = self.loader(entire_image_path)
		
		if self.transform is not None:
			context_image = self.transform(context_image)
			entire_image = self.transform(entire_image)	

		return context_image, entire_image

	def __len__(self):
		return len(self.imlist)

parser = argparse.ArgumentParser()
parser.add_argument('--mask',  default='segment', help='box | segment')
parser.add_argument('--ride',  default='horse', help='horse | bicycle')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--niter', type=int, default=1000, help='number of epochs to train for')

parser.add_argument('--testStep', type=int, default=100, help='test every testStep epoches')

parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64, help='#  of gen filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='#  of discrim filters in first conv layer')
parser.add_argument('--nc', type=int, default=3, help='# of channels in input')

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

# make test dir
exp_name = opt.mask+"_phone"
# if not exist, make dirs
result_test_dir = os.path.join("test_"+str(opt.arch), exp_name)
if not os.path.exists(result_test_dir):
	os.makedirs(result_test_dir)
else:
	clear_dir(result_test_dir)

test_log_dir = os.path.join(result_test_dir, 'log')
if not os.path.exists(test_log_dir):
	os.makedirs(test_log_dir)

result_train_dir = os.path.join("train_"+str(opt.arch), exp_name)

# redirect
current_time = time.time()
st = datetime.datetime.fromtimestamp(current_time).strftime('%Y-%m-%d_%H:%M:%S')
logfile = open(os.path.join(test_log_dir, 'test_'+st+'.txt'), 'w')
sys.stdout = Logger(sys.stdout, logfile)

# set datasets
transform = transforms.Compose([transforms.Scale((opt.imageSize,opt.imageSize)),
								transforms.ToTensor(),
								transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


triplet_dir = ""
test_file = "projection3_test.txt"

dataset = ImageContextDataset(os.path.join(triplet_dir, test_file), flist_reader=default_flist_reader, loader=default_image_loader, transform=transform)   
assert dataset

# data loader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers))

# loss function
criterionMSE = nn.MSELoss()

# network initialization
if opt.arch == "context":
	netG = _netG(opt)
	print('==> Generative model: context')
elif opt.arch == "res-patch":
	netG = _netG_resNet()
	print('==> Generative model: resNet')
elif opt.arch == "ushape":
	netG = UnetGenerator(3, 3, 7, opt.ngf)	

# start testing
start_epoch = 300
for epoch in list(range(start_epoch, opt.niter, opt.testStep))+[opt.niter-1]:
	cur_test_dir = os.path.join(result_test_dir, 'epoch_'+str(epoch))
	if not os.path.exists(cur_test_dir):
		os.makedirs(cur_test_dir)
	test_cropped_dir = os.path.join(cur_test_dir, 'cropped')
	if not os.path.exists(test_cropped_dir):
		os.makedirs(test_cropped_dir)
	test_real_dir = os.path.join(cur_test_dir, 'real')
	if not os.path.exists(test_real_dir):
		os.makedirs(test_real_dir)
	test_predict_dir = os.path.join(cur_test_dir, 'predict')
	if not os.path.exists(test_predict_dir):
		os.makedirs(test_predict_dir)

	# path to generative model
	netG_path = os.path.join(os.path.join(result_train_dir, 'model'), 'netG_epoch_%03d.pth' % (epoch))
	print('Loading '+netG_path)
	netG.load_state_dict(torch.load(netG_path, map_location=lambda storage, location: storage)['state_dict'])
	netG.eval()

        err_epoch = 0.

	# start test this epoch
	for i, (context_image, entire_image) in enumerate(dataloader):
		context_image_var = torch.autograd.Variable(context_image)
		entire_image_var = torch.autograd.Variable(entire_image)

		fake_image = netG(context_image_var)
		errG = criterionMSE(fake_image, entire_image_var)

		fake_image_numpy = fake_image.data.cpu().numpy()*0.5+0.5
		context_image_numpy = context_image_var.data.cpu().numpy()*0.5+0.5
		entire_image_numpy = entire_image_var.data.cpu().numpy()*0.5+0.5

		vutils.save_image(torch.from_numpy(entire_image_numpy), os.path.join(test_real_dir, 'real_%03d.png' % i))
		vutils.save_image(torch.from_numpy(context_image_numpy), os.path.join(test_cropped_dir, 'cropped_%03d.png' % i))
		vutils.save_image(torch.from_numpy(fake_image_numpy), os.path.join(test_predict_dir, 'predict_%03d.png' % i))

		print('[Epoch: %d/%d][Image: %d/%d] MSE_Loss: %.4f' % (epoch+1, opt.niter, i+1, len(dataloader), errG.data[0]))
                err_epoch += errG.data[0]
        
        err_epoch = err_epoch / len(dataloader)
        print('[Epoch: %d/%d] Average MSE_Loss: %.4f' % (epoch+1, opt.niter, err_epoch))
	print("==================================================================================================")	
