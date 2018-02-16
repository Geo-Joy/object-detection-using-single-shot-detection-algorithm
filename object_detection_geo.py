## Importing the required libraries

import torch #this contains PyTorch - best tool for Computer vision - as it contains the dynamic graphs - easier for gradient calculation
from torch.autograd import Variable # autograd is the module responsible for gradient decent
import cv2 #to draw rectangles on images
from data import BaseTransform, VOC_CLASSES as labelmap #data is a folder container BT and VC - BT does all image transforms making it compatible with neural network- VC is for encoding of classes eg: planes as 1, dogs as 2
from ssd import build_ssd #ssd is the library of SSD
import imageio #imageio library to process the images of the video


## Defining the function that will do the detections
# we will do a frame by frame detection
# this will work on each single images
# we use imageio to extract all frames from the video apply detect function and re-assemple the video.

def detect(frame, net, transform):
    #get the height, width 
    height, width = frame.shape[:2] #orframe.shape[:2]
    
    
    ##Original image to a Torch variable that can be accepted by SSD_NN
    # apply transform for right dimensions and frame color
    # convert this transformed frame from numpy array to torch_tensor
    # add a fake dimention to torch_tensorfor batch
    # convert it to a torch variable(both tensor and gradient)
    
    #1
    frame_transformed = transform(frame)[0]
    #2
    x = torch.from_numpy(frame_transformed).permute(2,0,1) # RGB to GRB
    print(x)
    #3 & 4
    x = Variable(x.unsqueeze(0)) #index for frst dimension
    #print(x)
    
    y = net(x)
    
    detections = y.data
    scale = torch.Tensor([width, height, width, height])
    
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.6:
            points = (detections[0, i, j, 1:] * scale).numpy()
            cv2.rectangle(frame, (int(points[0]), int(points[1])), (int(points[2]), int(points[3])), (255, 0, 0), 2)
            #print the label
            cv2.putText(frame, labelmap[i-1], (int(points[0]), int(points[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            j += 1
    return frame
            
##creating the SSD neural network
#creating NN object
net = build_ssd('test')
#load the weights from already pretrained NN
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage))

## creating the transformation
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/126.0))

## Doing object detection in video
reader = imageio.get_reader('funny_dog.mp4')# read the input video
fps = reader.get_meta_data()['fps'] # get the videos frame rate to get the number of frames to loop
writer = imageio.get_writer('output8.mp4', fps = fps) #append the processed images back to video
for i, frame in enumerate(reader): #append the processed images back to video
    processed_frame = detect(frame, net.eval(), transform) # net is an advances structure represents NN We call our detect function (defined above) to detect the object on the frame.
    writer.append_data(processed_frame); #append the processed frame back as a video
    print(i) # print the processed frame number
writer.close() #close the writer of image to video
    
    
    
