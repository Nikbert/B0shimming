import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import scipy.io as sio

import loaddata as ld


#def loaddata_siemens(file_path):
#    # Placeholder for the actual data loading function
#    # This function should load the data from the Siemens file and return it as a NumPy array
#    pass

# Echo times
echo_times = [2.22, 4.45]  # NW this should be read from the data or the pulseq file
data_path = '/media/wehkamp/data_store/myDataDir/shim_niels_cimaX_20240514/'
data_file_path = data_path + '2024-05-14-115610.dat'

# Load the data
data = ld.loaddata_siemens(data_file_path)

# Number of coils
n_coil = data.shape[3]
im_t1 = data[..., 0]
im_t2 = data[..., 1]

# Perform FFT operations
im_te1 = np.zeros_like(im_t1, dtype=np.complex_)
im_te2 = np.zeros_like(im_t2, dtype=np.complex_)

for ic in range(n_coil):
    im_te1[..., ic] = np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(im_t1[..., ic])))
    im_te2[..., ic] = np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(im_t2[..., ic])))

# Compute the images
img1 = np.sqrt(np.sum(np.abs(im_te1), axis=3))
img2 = np.sqrt(np.sum(np.abs(im_te2), axis=3))

# Show the images
#print('img1.shape', img1.shape)
#plt.figure()
#plt.imshow(img1[:, :, 30])
#plt.title('Sliced Data')
#plt.show()

# Save the NIfTI file
nib.save(nib.Nifti1Image(img1, np.eye(4)), 'sball.nii')

# Save the .mat file
sio.savemat(data_path + 'sball.mat', {'im_te1': im_te1, 'im_te2': im_te2, 'echotimes': echo_times})
print('saving sball done!')
