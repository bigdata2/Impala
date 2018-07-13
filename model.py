import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
from torchvision import transforms

def convLayer(in_planes, out_planes, kernel_size, stride):
    seq = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                  stride=stride, padding=1, bias=True),
        nn.LeakyReLU(True)
    )
    return seq

def fcLayer(input_features, output_features):
	seq = nn.Sequential (
	      nn.Linear(input_features, output_features),
              nn.LeakyReLU(True)
	)
	return seq

class model_A3C(nn.Module):
    def __init__(self, num_out_layers=16, num_channels=3, isActor=False):
	super(model_A3C, self).__init__()

	self.linear_input_dim = 2816 #output when image size is w=96,h=72
	self.linear_output_dim = 2 #Change to 256 when lstm is added
	self.isActor = isActor

	self.layer1 = convLayer(num_channels, num_out_layers, kernel_size=8, stride=4)
	self.layer2 = convLayer(num_out_layers, num_out_layers*2, kernel_size=4, stride=2)
	self.layer3 = fcLayer(self.linear_input_dim, self.linear_output_dim)
	self.lstm = nn.LSTMCell(256, 2)
	self.weights_init(self.layer1)
	self.weights_init(self.layer2)
	self.weights_init(self.layer3)

    def weights_init(self,module):
        for m in module.modules():
		if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                	init.xavier_uniform(m.weight, gain=np.sqrt(2))
                	init.constant(m.bias, 0)

    def forward(self, image_input):
	#image_input shape is (96, 72)
	normalize = transforms.Normalize(
        		mean = [127.5, 127.5, 127.5],
        		std  = [127.5, 127.5, 127.5]
	)
	preprocess = transforms.Compose([
			transforms.ToPILImage(),
   			transforms.Resize((96,72)),
   			transforms.ToTensor(),
   			normalize
	])
	img_tensor = preprocess(image_input)
	img_tensor.unsqueeze_(0)
        x = self.layer1(Variable(img_tensor)) if self.isActor else \
					self.layer1(Variable(img_tensor).cuda())
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
	return x
