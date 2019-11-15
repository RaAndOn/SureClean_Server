#************************************#
#**** 		Import Modules		 ****#
#************************************#

import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import dill

import sys
sys.path.insert(0, os.getcwd()+'/litter-detection/classifier_dl.py')
print sys.path[0]

#************************************#
#****   Global Variables		 ****#
#************************************#

model_file_name = os.path.join(os.getcwd(),'litter-detection/checkpoint_weights')

#************************************#
#**** 		MobileNet v2 		 ****#
#************************************#

class BasicBlock(nn.Module):
    def __init__(self, channel_size, stride=1, expansion_factor=6):
        super(BasicBlock, self).__init__()
        expansion_channel = expansion_factor*channel_size
        self.conv1 = nn.Conv2d(channel_size, expansion_channel, kernel_size=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(expansion_channel, expansion_channel, kernel_size=3, stride=stride, padding=1, groups=expansion_channel, bias=False)
        self.conv3 = nn.Conv2d(expansion_channel, channel_size, kernel_size=1, stride=1, bias=False)
        self.shortcut = nn.Conv2d(channel_size, channel_size, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu6(self.conv1(x))
        out = F.relu6(self.conv2(out))
        out = F.relu6(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu6(out)
        return out

#------  CNN Model with Residual Block #------  

class Network(nn.Module):
    def __init__(self, num_feats = 3, hidden_sizes = [32, 16, 64, 256, 1280],\
                 num_classes = 2, strides = [1, 2, 2, 1, 1], repeatitions = [1, 4, 3, 2], feat_dim=10):
        super(Network, self).__init__()
        self.hidden_sizes = [num_feats] + hidden_sizes + [num_classes]
        self.repeatitions = [1] + repeatitions
        self.strides = [2] + strides
        self.layers = []
        self.num_layers = len(self.hidden_sizes)
        
        # ** Hidden Layers ** 
        for idx in range(self.num_layers-2):

            
            if idx>0 and idx<self.num_layers-2:
                # ** All Bottle Neck Layers **
                repeat_count = 0
                while repeat_count < self.repeatitions[idx]:
                    self.layers.append(BasicBlock(channel_size=self.hidden_sizes[idx],expansion_factor=6))
                    repeat_count += 1
                
            # ** Convolutions **
            self.layers.append(nn.Conv2d(in_channels=self.hidden_sizes[idx], out_channels=self.hidden_sizes[idx+1], kernel_size=3, stride=self.strides[idx], padding=1, bias=False))

            self.layers.append(nn.BatchNorm2d(self.hidden_sizes[idx+1]))
            self.layers.append(nn.ReLU6(inplace=True))
          

        self.layers = nn.Sequential(*self.layers)
        self.linear_label = nn.Linear(self.hidden_sizes[-2], self.hidden_sizes[-1], bias=False)
        
        # For creating the embedding to be passed into the Center Loss criterion
        self.linear_closs = nn.Linear(self.hidden_sizes[-2], feat_dim, bias=False)
        self.relu_closs = nn.ReLU(inplace=True)
        # print(self.layers)

    
    def forward(self, x, evalMode=False):
        output = x
        output = self.layers(output)
            
        output = F.avg_pool2d(output, [output.size(2), output.size(3)], stride=1)
        output = output.reshape(output.shape[0], output.shape[1])
        embeddings = output
        
        label_output = self.linear_label(output)
        label_output = label_output/torch.norm(self.linear_label.weight, dim=1)

        if not evalMode:
            # Create the feature embedding for the Center Loss
            closs_output = self.linear_closs(output)
            closs_output = self.relu_closs(closs_output)
            return closs_output, label_output
        else:
            return output, label_output


def classify_image(net, image):
    input_img = [image]
    image = torch.stack(input_img)
    # print(image.size())
    feature, outputs = net(image)
    _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
    pred_labels = pred_labels.view(-1)
    return pred_labels.tolist()[0]   


def load_model():
    network = Network()
    print('Loading....' + str(model_file_name))
    # checkpoint = torch.load(model_file_name,map_location=torch.device('cpu'))
    # network = checkpoint['network']
    network.load_state_dict(torch.load(model_file_name))#checkpoint['network_state_dict'])
    # optimizer_label = checkpoint['optimizer_label']
    # optimizer_label.load_state_dict(checkpoint['optimizer_label_state_dict'])
    # optimizer_closs = checkpoint['optimizer_closs']
    # optimizer_closs.load_state_dict(checkpoint['optimizer_closs_state_dict'])
    network.eval()
    print('Success!! Loaded the pretrained model')
    return network


#************************************#
#****       Test Case            ****#
#************************************#
def test_cases(network, file_name):
    img = Image.open(file_name)
    img = torchvision.transforms.ToTensor()(img)
    print 'File:' + file_name + '| Class:' + str(classify_image(network, img))

if __name__ == '__main__':
    network = load_model()

    torch.save(network.state_dict(),'checkpoint_weights')

    relative_path = os.getcwd() + '/litter-detection/'
    test_cases(network,relative_path+'neg1.jpg')
    test_cases(network,relative_path+'pos1.jpg')
    test_cases(network,relative_path+'pos2.jpg')

