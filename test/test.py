from deepnet.deepclassify import SexyNet

net = SexyNet(img="aretha.png")
net.eval()
net.topprob()