from mapvbvd import mapVBVD as mapVBVD
from pypulseq.Sequence.read_seq import read
from pypulseq.Sequence.sequence import Sequence

import matplotlib.pyplot as plt
import numpy as np
#import phasecontrastmulticoil as pc

data_file_path = '/home/wehkamp/git/shim_test/2023-02-15-201504.dat' #baseline
data_file_path = '/home/wehkamp/git/shim_test/2023-02-15-210003.dat' #fn.z2.plus
data_file_path = '/home/wehkamp/git/shim_test/2023-02-15-203948.dat' #fn.z2.minus
#data_file_path = '/home/wehkamp/git/shim_test/2023-02-15-205625.dat' #fn.xy.plus
#data_file_path = '/home/wehkamp/git/shim_test/2023-02-15-205808.dat' #fn.xy.minus
#data_file_path = '/home/wehkamp/git/shim_test/2023-02-15-204250.dat' #fn.zx.plus
#data_file_path = '/home/wehkamp/git/shim_test/2023-02-15-204532.dat' #fn.zx.minus
#data_file_path = '/home/wehkamp/git/shim_test/2023-02-15-205151.dat' #fn.x2y2.plus
#data_file_path = '/home/wehkamp/git/shim_test/2023-02-15-205422.dat' #fn.x2y2.minus
#data_file_path = '/home/wehkamp/git/shim_test/2023-02-15-204734.dat' #fn.zy.plus
#data_file_path = '/home/wehkamp/git/shim_test/2023-02-15-204929.dat' #fn.zy.minus
#data_file_path = '/home/wehkamp/git/shim_test/2023-02-15-201930.dat' #fn.x.plus
#data_file_path = '/home/wehkamp/git/shim_test/2023-02-15-202059.dat' #fn.x.minus
#data_file_path = '/home/wehkamp/git/shim_test/2023-02-15-202248.dat' #fn.y.plus
#data_file_path = '/home/wehkamp/git/shim_test/2023-02-15-202421.dat' #fn.y.minus
#data_file_path = '/home/wehkamp/git/shim_test/2023-02-15-202557.dat' #fn.z.plus
#data_file_path = '/home/wehkamp/git/shim_test/2023-02-15-202836.dat' #fn.z.minus

""" load data (2 echoes interleaved) """
twixObj = mapVBVD(data_file_path,quiet=True)
twixObj.image.squeeze = True
data = twixObj.image['']
#data = twixObj.image[:,:,:]
#data = twixObj.image[0::2,0,:]
print('data.shape',data.shape)
img = np.abs(np.sqrt(np.sum(np.square(data), -1)))
print('img.shape',img.shape)
#plt.imshow(img)
#plt.show()


[gyroRatio] = twixObj.search_header_for_val('MeasYaps',('sTXSPEC', 'asNucleusInfo', '0', 'lFrequency')) #Hz/T

""" Retrieve the unsorted data """
twixObj.image.flagRemoveOS = False
data_unsorted = twixObj.image.unsorted()
print('data_unsorted.shape', data_unsorted.shape)

#img = np.abs(np.sqrt(np.sum(np.square(data_unsorted), -1)))
#plt.imshow(img)
#plt.show()

"""Attention!! hard coded Sequence Constants """
"""Jon s Matlab script"""
N = [60, 60, 60]
nx = N[0]
ny = N[1] 
nz = N[2]
nRead = 4*nx # 4x oversampling (will be fixed in future)
deltaTE = [0, 1000/440] # % echo-time difference (msec)

""" remove dummy shots """
nEcho = 2 #length(deltaTE) #attention hard coded shit!!!:w
nzDummy = 2 # see b04ge.m
nDummyShots = nzDummy * ny * nEcho
data = data_unsorted[: , : , (nDummyShots):] # [nFid nCoils 2*ny*nz]
print('data.shape',data.shape)
# 
# % separate the two images (different echo times)
dat1 = data[:, :, 0::2] # % short TE, [nRead nCoils ny*nz]
dat2 = data[:, :, 1::2] # % long TE
print('dat1.shape',dat1.shape)
print('dat2.shape',dat2.shape)
# 

#data_sorted = np.reshape(dat1, (sampling_points, rx_channels, n_rise_t_max, Nrep))
#240,2,3600
#plt.imshow(dat1[:,1,0:60])
#plt.show()

""" reshape to [nRead ny nz nCoils] """
nCoils = dat1.shape[1]
print('nCoils',nCoils)
dat1 = np.transpose(dat1, [0, 2, 1])
dat2 = np.transpose(dat2, [0, 2, 1])
dat1 = np.reshape(dat1, (nRead, ny, nz, nCoils))
dat2 = np.reshape(dat2, (nRead, ny, nz, nCoils))
print('dat1.shape', dat1.shape)

""" reconstruct """
im1 = np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(dat1)))
im2 = np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(dat2)))
print('im1.shape',im1.shape)

#plt.imshow(np.power(np.abs(im1[120,:,:,1]),0.2))
#plt.show()
print('done')

""" flip dimensions so the image aligns with Jon's 'universal coordinate system' (UCS) """
im1 = np.flip(im1, 1)
im1 = np.flip(im1, 3)
im2 = np.flip(im2, 1)
im2 = np.flip(im2, 3)

""" crop fov in x to account for Dwell time being fixed to 4us """
nc = round(nRead/2)  # center of image
print('nc',nc)
lob = int(nc-nx/2)
upb = int(nc+nx/2)
im1 = im1[lob:upb, :, :, :]
im2 = im2[lob:upb, :, :, :]
print('im1.shape',im1.shape)
 
#plt.figure(1) 
#plt.imshow(np.power(np.abs(im1[:,30,:,0]),0.2))
#plt.show()

""" coil-combined image """
#imsos1 = sqrt(sum(abs(im1).^2, 4))
imsos1 = np.sqrt(np.sum(np.power(np.abs(im1),2), 3))
imsos2 = np.sqrt(np.sum(np.power(np.abs(im2),2), 3))
imsos = imsos1 + imsos2
#plt.figure(2) 
#plt.imshow(imsos[:,30,:])
#plt.show()

""" object mask """
#mask = imsos > 0.2*max(imsos(:))
 
""" field map  """
#pc = pc.phasecontrastmulticoil(im2, im1)
pc = im1*im2.conj()
pc = np.angle(pc)

fmap = (pc/2/np.pi) / np.diff(deltaTE) * 1e3 # Hz
print('fmap.shape',fmap.shape)
#plt.figure(3) 
#plt.imshow(fmap[30,:,:,1])
#plt.title('magnitude sum-of-squares images')
#plt.colorbar()
#plt.show()

mask = np.ma.masked_outside(fmap, -40, 40)

#plt.figure(4) 
#plt.imshow(mask[30,:,:,1],'RdBu')
#plt.title('magnitude sum-of-squares images')
#plt.colorbar()
#plt.show()


# Calculate the distance from the center of the circle
ny, nx, nz, channels = fmap.shape
print('ny',ny)
ix, iy = np.meshgrid(np.arange(nx), np.arange(ny))
center_x = nx/2
center_y = ny/2
center_z = nz/2
distance = np.sqrt((ix - center_x)**2 + (iy - center_y)**2)

radius = 17 ### Attention !!!!!!! 
## Mask portions of the data array outside of the circle
fmap_mask = np.ma.masked_where(distance > radius, mask[:,:,30,1])

""" Display magnitude images and field map. """
plt.figure(4) 
plt.imshow(fmap_mask,'RdBu', vmin=-30, vmax=30)
plt.title('fieldmap in [Hz]')
plt.colorbar()
plt.show()
