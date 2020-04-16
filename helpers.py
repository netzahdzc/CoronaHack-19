# helper functions for saving sample data and models
import os
import pdb
import pickle
import argparse
import imageio
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import scipy
import scipy.misc


def checkpoint(iteration, G_XtoY, G_YtoX, D_X, D_Y, checkpoint_dir='checkpoints_cyclegan'):
    G_XtoY_path = os.path.join(checkpoint_dir, 'G_XtoY.pkl')
    G_YtoX_path = os.path.join(checkpoint_dir, 'G_YtoX.pkl')
    D_X_path = os.path.join(checkpoint_dir, 'D_X.pkl')
    D_Y_path = os.path.join(checkpoint_dir, 'D_Y.pkl')
    torch.save(G_XtoY.state_dict(), G_XtoY_path)
    torch.save(G_YtoX.state_dict(), G_YtoX_path)
    torch.save(D_X.state_dict(), D_X_path)
    torch.save(D_Y.state_dict(), D_Y_path)


def merge_images(sources, targets, batch_size=16):
    _, _, h, w = sources.shape
    row = int(np.sqrt(batch_size))
    merged = np.zeros([3, row*h, row*w*2])
    for idx, (s, t) in enumerate(zip(sources, targets)):
        i = idx // row
        j = idx % row
        merged[:, i*h:(i+1)*h, (j*2)*h:(j*2+1)*h] = s
        merged[:, i*h:(i+1)*h, (j*2+1)*h:(j*2+2)*h] = t
    merged = merged.transpose(1, 2, 0)
    return merged
    

def to_data(x):
    if torch.cuda.is_available():
        x = x.cpu()
    x = x.data.numpy()
    x = ((x +1)*255 / (2)).astype(np.uint8) # rescale to 0-255
    return x

def save_samples(iteration, fixed_Y, fixed_X, G_YtoX, G_XtoY, batch_size=16, sample_dir='samples_cyclegan'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    fake_X = G_YtoX(fixed_Y.to(device))
    fake_Y = G_XtoY(fixed_X.to(device))
    
    X, fake_X = to_data(fixed_X), to_data(fake_X)
    Y, fake_Y = to_data(fixed_Y), to_data(fake_Y)

    merged = merge_images(X, fake_Y, batch_size)
    path = os.path.join(sample_dir, 'sample-{:06d}-X-Y.png'.format(iteration))
    img_uint8 = merged.astype(np.uint8)
    imageio.imwrite(path, img_uint8)
    print('Saved {}'.format(path))
    
    merged = merge_images(Y, fake_X, batch_size)
    path = os.path.join(sample_dir, 'sample-{:06d}-Y-X.png'.format(iteration))
    img_uint8 = merged.astype(np.uint8)
    imageio.imwrite(path, img_uint8)
    print('Saved {}'.format(path))
    
def save_images(iteration, fixed_Y, fixed_X, G_YtoX, G_XtoY, batch_size=16, sample_dir='fake_images_cyclegan', option='fake'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    fake_X = G_YtoX(fixed_Y.to(device))
    fake_Y = G_XtoY(fixed_X.to(device))
    
    _, fake_X = to_data(fixed_X), to_data(fake_X)
    _, fake_Y = to_data(fixed_Y), to_data(fake_Y)

    if option == 'fake':
        i = 1
        for fake in fake_Y: 
            path = os.path.join(sample_dir, 'sample-{:06d}-X-Y.png'.format(i))
            img_uint8 = fake.astype(np.uint8)
            imageio.imwrite(path, img_uint8)
            print('Fake saved {}'.format(path))
            i += 1

    if option == 'original':
        i = 1
        for fake in fake_X: 
            path = os.path.join(sample_dir, 'sample-{:06d}-Y-X.png'.format(i))
            img_uint8 = merged.astype(np.uint8)
            imageio.imwrite(path, img_uint8)
            print('Original saved {}'.format(path))
            i += 1
