import torch
from models import *
from dataIO import *
from test_train import *
import os

if __name__ == "__main__":
    epoch = 10000
    k = 3
    l = 1
    lr = 1e-4
    drop_prob = 0.2
    batch_size = 50
    z_size = 64
    img_size = (28, 28)
    mode = 'test' #train, test

    model = Model(img_size, z_size, batch_size, drop_prob)

    if mode == 'train':
        opt_gen = torch.optim.AdamW(model.generator.parameters(), lr)
        opt_dis = torch.optim.AdamW(model.discriminator.parameters(), lr)

        if os.path.isfile('model_data.pth'): model.load_state_dict(torch.load('model_data.pth'))

        for ep in range(epoch):
            print("epoch ", ep)
            img = get_mnist_img(batch_size)
            img = torch.reshape(img, (-1, img_size[0]*img_size[1]))
            z = get_z(batch_size, z_size)
            train(k, l, model, opt_gen, opt_dis, img, z)

        torch.save(model.state_dict(), 'model_data.pth')

    elif mode == 'test':
        z = get_z(batch_size, z_size)
        model.load_state_dict(torch.load('model_data3_1_10000.pth'))
        make_sample_img(model, z, z_size)