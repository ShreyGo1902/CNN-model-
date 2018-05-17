from copy import deepcopy
class OcclusionGenerator(object):
     
    def __init__(self, img, boxsize=8, step=8):
         '''Initializations '''
         self.img = img
         self.boxsize = boxsize
         self.step = step 
         self.i = 0
         self.j = 0
    
    def flow(self):
        '''Return a single occluded image and its location'''
        if self.i + self.boxsize > self.img.shape[0]:
            return None, None, None
        
        retImg = np.copy(self.img)
        retImg[self.i:self.i+self.boxsize, self.j:self.j+self.boxsize] = 0.0 

        old_i = deepcopy(self.i) 
        old_j = deepcopy(self.j)
        
        # update indices
        self.j = self.j + self.step
        if self.j+self.boxsize>self.img.shape[1]: #reached end
            self.j = 0 # reset j
            self.i = self.i + self.step # go to next row
        
        return retImg, old_i, old_j

    def gen_minibatch(self,batchsize=64):
        '''Returns a minibatch of images of size <=batchsize '''

        # list of occluded images
        occ_imlist = []
        locations = []
        for i in range(batchsize):
            occimg,i,j = self.flow()
            if occimg is not None:
                occ_imlist.append(occimg.astype('float32'))
                locations.append([i,j])
        if len(occ_imlist)==0:
            return None,None
        else:
            #occ_imlist = np.asarray(occ_imlist)
            mean = []
            sd = []

            for cnt,i in enumerate(occ_imlist):
                mean.append(cv2.meanStdDev(i)[0])
                sd.append(cv2.meanStdDev(i)[1])
            mean1 = np.zeros((3,1))
            sd1 = np.zeros((3,1))
            mean1 = sum(mean)/len(mean)
            sd1 = sum(sd)/len(sd)
            for cnt in range(len(occ_imlist)):
                i = occ_imlist[cnt]
                i[:,:,0] = (i[:,:,0] - mean1[0][0])/sd1[0][0]
                i[:,:,1] = (i[:,:,1] - mean1[1][0])/sd1[1][0]
                i[:,:,2] = (i[:,:,2] - mean1[2][0])/sd1[2][0]
                occ_imlist[cnt] = i
            occ_imlist = np.asarray(occ_imlist)
            print('0',occ_imlist[0])
            print('1',occ_imlist[1])
            return occ_imlist , locations
def post_process(heatmap):

	# postprocessing
	total = heatmap[0]
	for val in heatmap[1:]:
		total = total + val

       
	return total


def gen_heat_map():
        a = malignant_train[7]
        occ = OcclusionGenerator(a, 4, 4)
        heatmap = []
        index = 0

        x, locations = occ.gen_minibatch(batchsize=256)
        x = x.reshape(x.shape[0],3,64,64)

        op = model.predict(x)

        for i in range(x.shape[0]):
            score = op[i][1]
            r,c = locations[i] 
            scoremap = np.zeros((64,64))
            scoremap[r : r+occ.boxsize, c : c+occ.boxsize] = score
            heatmap.append(scoremap)
        
        return heatmap
heatmap = gen_heat_map()
processed = post_process(heatmap)

plt.imshow(processed,cmap='hot',interpolation='nearest')
plt.show()
'''

model.layers[0].get_weights()
gives the weigths of the filters
a[weigths,biasis][no of filters][row length][hiegth length]


weigths = model.layers[0].get_weights() # [weights,biasis][no of filters][length][heigth]

for i in range(32):
    a1 = weigths[0][i]
    a1 = a1.reshape(5,5).astype('float32')
    plt.subplot(8,4,i+1)
    plt.imshow(a1,cmap = plt.get_cmap('gray'))

plt.show()

model1 = Sequential()
model1.add(Convolution2D(32, 5, 5,input_shape=(1, 28, 28),activation= 'relu', weights=model.layers[0].get_weights()))
activations = model1.predict(X_train[:1])[0]#activations == 64000,32,24,24 # X_train ==60000,1,28,28

for i in range(32):
    a1 = activations[i]
    plt.subplot(8,4,i+1)
    plt.imshow(a1,cmap = plt.get_cmap('gray'))

plt.show()    



# with a Sequential model for layer Viszualiztion
get_1st_layer_output = K.function([model.layers[0].input],
                                  [model.layers[0].output])
layer_output = get_1st_layer_output([X_train[:1]])[0]
for i in range(32):
    a1 = layer_output[0][i]
    plt.subplot(8,4,i+1)
    plt.imshow(a1,cmap = plt.get_cmap('gray'))
'''
