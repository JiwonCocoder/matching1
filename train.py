from __future__ import print_function, division
import os
from os.path import exists, join, basename
from collections import OrderedDict
import numpy as np
import numpy.random
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import relu
from torch.utils.data import Dataset

from lib.dataloader import DataLoader # modified dataloader
from lib.model import ImMatchNet
from lib.im_pair_dataset import ImagePairDataset
from lib.normalization import NormalizeImageDict, UnNormalize
from lib.torch_util import save_checkpoint, str_to_bool
from lib.torch_util import BatchTensorToVars, str_to_bool
from image.normalization import NormalizeImageDict, normalize_image

import argparse

from torchvision import transforms
import torch
from matplotlib import pyplot as plt
from skimage import io,data

# Seed and CUDA
use_cuda = torch.cuda.is_available()
torch.manual_seed(1)
if use_cuda:
    torch.cuda.manual_seed(1)
np.random.seed(1)

print('ImMatchNet training script')

# Argument parsing
parser = argparse.ArgumentParser(description='Compute PF Pascal matches')
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--image_size', type=int, default=400)
parser.add_argument('--dataset_image_path', type=str, default='datasets/pf-pascal/', help='path to PF Pascal dataset')
parser.add_argument('--dataset_csv_path', type=str, default='datasets/pf-pascal/image_pairs/', help='path to PF Pascal training csv')
parser.add_argument('--num_epochs', type=int, default=5, help='number of training epochs')
parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--ncons_kernel_sizes', nargs='+', type=int, default=[5,5,5], help='kernels sizes in neigh. cons.')
parser.add_argument('--ncons_channels', nargs='+', type=int, default=[16,16,1], help='channels in neigh. cons')
parser.add_argument('--result_model_fn', type=str, default='checkpoint_adam', help='trained model filename')
parser.add_argument('--result-model-dir', type=str, default='trained_models', help='path to trained models folder')
parser.add_argument('--fe_finetune_params',  type=int, default=0, help='number of layers to finetune')


args = parser.parse_args()
print(args)

# Create model
print('Creating CNN model...')
model = ImMatchNet(use_cuda=use_cuda,
				   checkpoint=args.checkpoint,
                   ncons_kernel_sizes=args.ncons_kernel_sizes,
                   ncons_channels=args.ncons_channels)

# Set which parts of the model to train
if args.fe_finetune_params>0:
    for i in range(args.fe_finetune_params):
        for p in model.FeatureExtraction.model[-1][-(i+1)].parameters(): 
            p.requires_grad=True

print('Trainable parameters:')
for i,p in enumerate(filter(lambda p: p.requires_grad, model.parameters())): 
    print(str(i+1)+": "+str(p.shape))
    
# Optimizer
print('using Adam optimizer')
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
cnn_image_size=(args.image_size,args.image_size)

Dataset = ImagePairDataset
train_csv = 'train_pairs.csv'
test_csv = 'val_pairs.csv'
normalization_tnf = NormalizeImageDict(['source_image','target_image'])
batch_preprocessing_fn = BatchTensorToVars(use_cuda=use_cuda)   

# Dataset and dataloader
dataset = Dataset(transform=normalization_tnf,
	              dataset_image_path=args.dataset_image_path,
	              dataset_csv_path=args.dataset_csv_path,
                  dataset_csv_file = train_csv,
                  output_size=cnn_image_size)

dataloader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=0)

dataset_test = Dataset(transform=normalization_tnf,
	                   dataset_image_path=args.dataset_image_path,
	                   dataset_csv_path=args.dataset_csv_path,
                       dataset_csv_file=test_csv,
                       output_size=cnn_image_size)

dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size,
                        shuffle=True, num_workers=4)
    
# Define checkpoint name
checkpoint_name = os.path.join(args.result_model_dir,
                               datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")+'_'+args.result_model_fn + '.pth.tar')

print('Checkpoint name: '+checkpoint_name)    
    
# Train
best_test_loss = float("inf")

def weak_loss(model,batch,normalization='softmax',alpha=30):
    if normalization is None:
        normalize = lambda x: x
    elif normalization=='softmax':     
        normalize = lambda x: torch.nn.functional.softmax(x,1)

    elif normalization=='l1':
        normalize = lambda x: x/(torch.sum(x,dim=1,keepdim=True)+0.0001)

    b = batch['source_image'].size(0)
    # positive
    #corr4d = model({'source_image':batch['source_image'], 'target_image':batch['target_image']})
    corr4d = model(batch)
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    # print(corr4d.shape)
    # corr4d_torch = torch.tensor(corr4d).float()
    # inv_corr4d = inv_normalize(batch['source_image'][0])
    # print(inv_corr4d)
    source_image = normalize_image(batch['source_image'], forward=False)
    source_image = source_image.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy()
    print("source_image_shape")
    print(batch['source_image'].shape, source_image.shape)
    print(source_image)
    warped_image = normalize_image(corr4d, forward=False)
    warped_image = warped_image.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy()
    print("warped_image_shape")
    print(batch['target_image'].shape, warped_image.shape)
    print(warped_image)
    target_image = normalize_image(batch['target_image'], forward=False)
    target_image = target_image.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy()
    # check if display is available
    exit_val = os.system('python -c "import matplotlib.pyplot as plt;plt.figure()"  > /dev/null 2>&1')
    display_avail = exit_val == 0

    if display_avail:
        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(source_image)
        axs[0].set_title('src')
        axs[1].imshow(warped_image)
        axs[1].set_title('warped')
        axs[2].imshow(target_image)
        axs[2].set_title('tgt')

        print('Showing results. Close figure window to continue')
        plt.show()


loss_fn = lambda model,batch: weak_loss(model,batch,normalization='softmax')

# define epoch function
def process_epoch(mode,epoch,model,loss_fn,optimizer,dataloader,batch_preprocessing_fn,use_cuda=True,log_interval=50):
    epoch_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        if mode=='train':            
            optimizer.zero_grad()
        tnf_batch = batch_preprocessing_fn(batch)
        loss_fn(model,tnf_batch)

    if mode=='train':
        epoch_loss = None

    return epoch_loss

train_loss = np.zeros(args.num_epochs)
test_loss = np.zeros(args.num_epochs)

print('Starting training...')

model.FeatureExtraction.eval()

for epoch in range(1, args.num_epochs+1):
    train_loss[epoch-1] = process_epoch('train',epoch,model,loss_fn,optimizer,dataloader,batch_preprocessing_fn,log_interval=1)
    test_loss[epoch-1] = process_epoch('test',epoch,model,loss_fn,optimizer,dataloader_test,batch_preprocessing_fn,log_interval=1)
      
    # remember best loss
    is_best = test_loss[epoch-1] < best_test_loss
    best_test_loss = min(test_loss[epoch-1], best_test_loss)
    save_checkpoint({
        'epoch': epoch,
        'args': args,
        'state_dict': model.state_dict(),
        'best_test_loss': best_test_loss,
        'optimizer' : optimizer.state_dict(),
        'train_loss': train_loss,
        'test_loss': test_loss,
    }, is_best,checkpoint_name)

print('Done!')
