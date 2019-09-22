import numpy as np


class Conv3x3:
    def __init__(self,num_filters):
        self.num_filter = num_filters
        self.filters = np.random.randn(num_filters,3,3) / 9 #xavier initialization

    def iterate_regions(self,image):
        '''
        padding is valid and image is 2d numpy array
        '''
        h,w = image.shape

        for i in range(h-2):
            for j in range(w-2):
                im_region = image[i:(i+3),j:(j+3)]
                # print(im_region)
                yield im_region,i,j

    def forward(self,input):
        '''
        performs a forward pass on a given input.
        Returns a 3D array having the shape as  (h,w,num_filters)
        '''
        h,w = input.shape
        output= np.zeros((h-2,w-2,self.num_filter))

        for im_region,i,j in self.iterate_regions(input):
            output[i,j] = np.sum(im_region*self.filters,axis=(1,2))

        return output

if __name__ == "__main__":
    Conv = Conv3x3(5)
    img= np.random.randn(28,28)
    print('Conv class has been created')
    output = Conv.forward(img)
    print('Forward pass done on one image')