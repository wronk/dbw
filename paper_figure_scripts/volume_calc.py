'''
@author wronk

Script to calculate the volume of the mouse brain from the common coordinate
frame developed by ABI. This requires the connectivity dataset and the
allensdk.

Output from connectivity dataset (as of Sept 2015) is below.
'''

import numpy as np
from allensdk.core.mouse_connectivity_cache import (MouseConnectivityCache as
                                                    MCC)

# load in nrdd data
mcc = MCC(manifest_file='/Volumes/Brain2015/connectivity/manifest.json')
annot, annot_info = mcc.get_annotation_volume()

# get all voxels that are non-zeros
brain_voxels = annot.flatten() > 0
sum_vox = np.sum(brain_voxels)

print 'Found %d voxels' % (sum_vox)
print 'Density is %0.3f percent' % (100 * sum_vox / len(brain_voxels))

# Get size of each box, make sure they're in cubes
width_dims = []
for di in range(3):
    width_dims.append(annot_info['space directions'][di][di])
assert len(set(width_dims)) == 1

# width_dims in um, convert to cm, get cm^3
vol_per_box = ((width_dims[0] * 1e-6) * 1e2) ** 3

# multiply them by volume of each box
vol = sum_vox * vol_per_box
print 'Volume of mouse brain is: %0.3f cm^3' % (vol)

# OUTPUT (confirmed as reasonable by Lydia)
# Found 31907373 voxel
# Density is 41.000 percent
# Volume of the mouse brain is 0.499 cm^3
