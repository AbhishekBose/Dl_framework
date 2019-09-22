import numpy as np

class MaxPool:
    def __init__(self,size):
        self.size = size

    def iterate_regions(self,image):
        '''
        Non overalapping pooling over images happens
        '''
        h,w = image.shape
        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i*2):(i*2+2),(j*2):(j*2+2)]
                yield im_region,i,j

    def forward(self,input):
        '''
        performs forward pass with max pool
        Returns 3d numpy array with dimensions (h/2,w/2,num_filters)
        input is 3d numpy array with dimension (h,w,num_filters)
        '''
        h,w,num_filters = input.shape
        output = np.zeros((h // 2, w // 2, num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i,j] = np.max(im_region,axis=(0,1))

        return output