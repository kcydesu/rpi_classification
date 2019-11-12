from deepnet.deepclassify import SexyNet

net = SexyNet(img="aretha.jpg")
net.eval()
net.topprob()