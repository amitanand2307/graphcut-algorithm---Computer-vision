# -*- coding: utf-8 -*-
import maxflow
from numpy import *
import numpy
from pylab import *
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import scipy


def my_imshow(im, title=None, **kwargs):
    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'gray'
    plt.figure()
    plt.imshow(im, interpolation='none', **kwargs)
    if title:
        plt.title(title)
    plt.axis('off')


def graphcut(input_file, sigma, kappa, foreground, background):
# input image, Sigma value --> determines how fast the values decay towards zero with increasing dissimilarity, kappa value --> similar pixels have weight close to kappa, # foreground area, background area 
    Img = (Image.open(input_file).convert('L')) 
    Ifore = Img.crop(foreground) # take a part of the foreground
    Iback = Img.crop(background) # take a part of the background
    Img,Ifore,Iback = array(Img),array(Ifore),array(Iback) # convert all the images to arrays to calculation
    Iforemean,Ibackmean = mean(cv2.calcHist([Ifore],[0],None,[256],[0,256])),mean(cv2.calcHist([Iback],[0],None,[256],[0,256])) #Taking the mean of the histogram
    F,B =  ones(shape = Img.shape),ones(shape = Img.shape) #initalizing the foreground/background probability vector
    Im = Img.reshape(-1,1) #Coverting the image array to a vector for ease.
    m,n = Img.shape[0],Img.shape[1]# copy the size
    g,pic = maxflow.Graph[int](m,n),maxflow.Graph[int]() # define the graph
    structure = np.array([[inf, 0, 0],
                          [inf, 0, 0],
                          [inf, 0, 0]
                         ]) # initializing the structure....
    source,sink,J = m*n,m*n+1,Img # Defining the Source and Sink (terminal)nodes.
    nodes,nodeids = g.add_nodes(m*n),pic.add_grid_nodes(J.shape) # Adding non-nodes
    pic.add_grid_edges(nodeids,0),pic.add_grid_tedges(nodeids, J, 255-J)
    gr = pic.maxflow()
    IOut = pic.get_grid_segments(nodeids)
    for i in range(Img.shape[0]): # Defining the Probability function....
        for j in range(Img.shape[1]):
            F[i,j] = -log(abs(Img[i,j] - Iforemean)/(abs(Img[i,j] - Iforemean)+abs(Img[i,j] - Ibackmean))) # Probability of a pixel being foreground
            B[i,j] = -log(abs(Img[i,j] - Ibackmean)/(abs(Img[i,j] - Ibackmean)+abs(Img[i,j] - Iforemean))) # Probability of a pixel being background
    F,B = F.reshape(-1,1),B.reshape(-1,1) # convertingb  to column vector for ease
    for i in range(Im.shape[0]):
        Im[i] = Im[i] / linalg.norm(Im[i]) # normalizing the input image vector 
    w = structure # defining the weight       
    for i in range(m*n):#checking the 4-neighborhood pixels
        ws=(F[i]/(F[i]+B[i])) # source weight
        wt=(B[i]/(F[i]+B[i])) # sink weight
        g.add_tedge(i,ws[0],wt) # edges between pixels and terminal
        if i%n != 0: # for left pixels
            w = kappa*exp(-(abs(Im[i]-Im[i-1])**2)/sigma) # the cost function for two pixels
            g.add_edge(i,i-1,w[0],kappa-w[0]) # edges between two pixels
            '''The likelihood function: 
			* used Bayes’ theorem for conditional probabilities
            * The function is constructed by multiplying the individual conditional probabilities of a pixel being either 
            foreground or background in order to get the total probability. Then the class with highest probability is selected.
            * for a pixel i in the image:
                               * weight from sink to i:
                               probabilty of i being background/sum of probabilities
                               * weight from source to i:
                               probabilty of i being foreground/sum of probabilities
                               * weight from i to a 4-neighbourhood pixel:
                                K * e−|Ii−Ij |2 / s
                                 where k and s are parameters that determine hwo close the neighboring pixels are how fast the values
                                 decay towards zero with increasing dissimilarity
            '''
        if (i+1)%n != 0: # for right pixels
            w = kappa*exp(-(abs(Im[i]-Im[i+1])**2)/sigma)
            g.add_edge(i,i+1,w[0],kappa-w[0]) # edges between two pixels
        if i//n != 0: # for top pixels
            w = kappa*exp(-(abs(Im[i]-Im[i-n])**2)/sigma)
            g.add_edge(i,i-n,w[0],kappa-w[0]) # edges between two pixels
        if i//n != m-1: # for bottom pixels
            w = kappa*exp(-(abs(Im[i]-Im[i+n])**2)/sigma)
            g.add_edge(i,i+n,w[0],kappa-w[0]) # edges between two pixels
    Img = array(Image.open(input_file)) # calling the input image again to ensure proper pixel intensities....
    print "The maximum flow for %s is %d"%(input_file,gr) # find and print the maxflow
    Iout = ones(shape = nodes.shape)
    for i in range(len(nodes)):
        Iout[i] = g.get_segment(nodes[i]) # classifying each pixel as either forground or background
    out = 255*ones((Img.shape[0],Img.shape[1],3)) # initialization for 3d input
    
    for i in range(Img.shape[0]):
        for j in range(Img.shape[1]): # converting the True/False to Pixel intensity
            if IOut[i,j]==False:
                out[i,j,0],out[i,j,1],out[i,j,2] = Img[i,j,0],Img[i,j,1],Img[i,j,2] # foreground       
            else:
                out[i,j,0],out[i,j,1],out[i,j,2] = 1,255,255 # background
            
    figure()
    plt.imshow(out,vmin=0,vmax=255) # save the output image
    #plt.show()
    savefig('amit.png')
    file_in = "amit.png"
    img = Image.open(file_in)
    file_out = "amit.bmp"
    img.save(file_out)
    # Read the image you want connected components of
    src = cv2.imread("amit.bmp", 0)
    #binary_map = (src > 0).astype(np.uint8)
    
    # Threshold it so it becomes binary
    ret, thresh = cv2.threshold(src,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    connectivity = 4  
    
    output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
    
    # The first cell is the number of labels
    num_labels = output[0]
    print "The number of cars is:", num_labels,
    # The second cell is the label matrix
    labels = output[1]
    #print labels,
    # The third cell is the stat matrix
    stats = output[2]
    #print stats
    # The fourth cell is the centroid matrix
    centroids = output[3]
    #print centroids
    
    #print connected_components(out)
    
'''
    #pixels = numpy.zeros((200, 300, 3), dtype=numpy.uint8)
    #numpy.asarray(out)
    #print type(out)
    #im = Image.fromarray(out)
    #import numpngw
    #out.astype(uint8)
    #numpngw.write_png('foo.png', out, bitdepth=1)

    import png
    zgray = out[:, :, 0]
    with open('foo_color.png', 'wb') as f:
        writer = png.Writer(width=out.shape[1], height=out.shape[0], bitdepth=16, greyscale=True)
        # Convert z to the Python list of lists expected by
        # the png writer.
        zgray2list = zgray.tolist()
        writer.write(f, zgray2list)
    
    #im.show()
    #print out.shape()
    #print count1, count2

    img = Image.fromarray(out, 'RGB')
    img.save('my.jpg')
    #img.show()
'''
   
    

graphcut('input1.jpg',100,2,(225,142,279,185),(7,120,61,163)) #calling the maxflow funtion for input1
print "\n"
graphcut('input2.jpg',120,2,(148,105,201,165),(11,12,80,52)) #calling the maxflow funtion for input2
