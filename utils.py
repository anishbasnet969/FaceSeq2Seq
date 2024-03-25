import torch.nn as nn
import matplotlib.pyplot as plt


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def plot_images(images):
    x = images["input"]
    reconstruction = images["reconstruction"]

    fig, ax = plt.subplots(1, 2)

    # print(x[0].cpu().detach().numpy()[0].shape)
    # print(reconstruction[0].cpu().detach().numpy()[0].shape)

    ax[0].imshow(x.cpu().detach().numpy()[1].transpose(1, 2, 0))
    ax[1].imshow(reconstruction.cpu().detach().numpy()[1].transpose(1, 2, 0))
    plt.show()
