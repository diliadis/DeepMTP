import torch
from torch import nn
import torchvision.models as models
from collections import OrderedDict

class MLP(nn.Sequential):
	''' A standard fully connected feed-forward neural network.
	'''	
	def __init__(self, config, input_dim, output_dim, nodes_per_layer, num_layers, dropout_rate, batch_norm):
		super(MLP, self).__init__()
		if isinstance(nodes_per_layer, list):
			num_layers = len(nodes_per_layer)
			dims = [input_dim] + nodes_per_layer + [output_dim]
		else:
			dims = [input_dim] + num_layers*[nodes_per_layer] + [output_dim] 
		# self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(num_layers+1)])
		self.predictor = nn.ModuleList()
		for i in range(num_layers+1):
			self.predictor.append(nn.Linear(dims[i], dims[i+1]))
			self.predictor.append(nn.LeakyReLU())
			if i != num_layers:
				if batch_norm:
					# pass
					self.predictor.append(nn.BatchNorm1d(dims[i+1]))
				if dropout_rate != 0:
					self.predictor.append(nn.Dropout(dropout_rate))

	def forward(self, v):
		# predict
		v = v.float()
		for i, layer in enumerate(self.predictor):
			v = layer(v)
		return v  

class ConvNet(nn.Sequential):
	''' A convolutional neural network that is based on resnet.
	'''	
	def __init__(self, config, input_dim, output_dim, conv_architecture, conv_architecture_version, conv_architecture_last_trained_layer ,conv_architecture_dense_layers):
		super(ConvNet, self).__init__()

		self.predictor = nn.ModuleList()
		
		if conv_architecture == 'resnet':
			in_size = 512			
			if conv_architecture_version == 'resnet18':
				self.predictor.append(models.resnet18(pretrained=True))
				in_size = 512
			elif conv_architecture_version == 'resnet101':
				self.predictor.append(models.resnet101(pretrained=True))
				in_size = 2048
			else:
				raise AttributeError('The '+conv_architecture_version+' version of resnet is not currently supported')

			for param in self.predictor.parameters():
				param.requires_grad = False

			if conv_architecture_last_trained_layer == "last":
				pass
			elif conv_architecture_last_trained_layer == "layer4":
				for param in self.predictor[0].avgpool.parameters():
					param.requires_grad = True
				for param in self.predictor[0].layer4.parameters():
					param.requires_grad = True
			elif conv_architecture_last_trained_layer == "layer3":
				for param in self.predictor[0].avgpool.parameters():
					param.requires_grad = True
				for param in self.predictor[0].layer4.parameters():
					param.requires_grad = True
				for param in self.predictor[0].layer3.parameters():
					param.requires_grad = True
			elif conv_architecture_last_trained_layer == "layer2":
				for param in self.predictor[0].avgpool.parameters():
					param.requires_grad = True
				for param in self.predictor[0].layer4.parameters():
					param.requires_grad = True
				for param in self.predictor[0].layer3.parameters():
					param.requires_grad = True
				for param in self.predictor[0].layer2.parameters():
					param.requires_grad = True
			elif conv_architecture_last_trained_layer == "layer1":
				for param in self.predictor[0].avgpool.parameters():
					param.requires_grad = True
				for param in self.predictor[0].layer4.parameters():
					param.requires_grad = True
				for param in self.predictor[0].layer3.parameters():
					param.requires_grad = True
				for param in self.predictor[0].layer2.parameters():
					param.requires_grad = True
				for param in self.predictor[0].layer1.parameters():
					param.requires_grad = True
			else:
				raise AttributeError('Layer: '+str(conv_architecture_last_trained_layer)+' layer is not a valid '+str(conv_architecture_version)+' layer')

			if conv_architecture_dense_layers == 1:
				fc = nn.Sequential(
					OrderedDict(
						[('fc1', nn.Linear(in_size, output_dim)),]
					)
				)
			elif conv_architecture_dense_layers == 2:
				fc = nn.Sequential(
					OrderedDict(
						[
							("fc1", nn.Linear(in_size, in_size)),
							("relu", nn.LeakyReLU()),
							("fc2", nn.Linear(in_size, output_dim)),
						]
					)
				)
			else:
				raise AttributeError('Only 1 or 2 dense layers are currently allowed in the '+str(conv_architecture_version)+' version of resnet')

			self.predictor[0].fc = fc

		elif conv_architecture == 'VGG':

			self.predictor.append(models.vgg11_bn(pretrained=True))
			for param in self.predictor.parameters():
				param.requires_grad = False

			classifier = nn.Sequential(
				nn.Linear(in_features=25088, out_features=4096),
				nn.ReLU(inplace=True),
				nn.Dropout(p=0.5, inplace=False),
				nn.Linear(in_features=4096, out_features=4096),
				nn.ReLU(inplace=True),
				nn.Dropout(p=0.5, inplace=False),
				nn.Linear(in_features=4096, out_features=output_dim),
				nn.ReLU(inplace=True),
			)
			self.predictor[0].classifier = classifier

	def forward(self, v):
		# predict
		v = v.float()
		for i, layer in enumerate(self.predictor):
			v = layer(v)
		return v  

		