import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

def train(k, l, model, optimizer_gen, optimizer_dis, img, z):
    for _ in range(k):
        loss_dis = model.loss_dis(img, z)
        optimizer_dis.zero_grad()
        loss_dis.backward()
        optimizer_dis.step()
        print("loss_dis = %f"%loss_dis)
    for _ in range(l):
        loss_gen = model.loss_gen(z)
        optimizer_gen.zero_grad()
        loss_gen.backward()
        optimizer_gen.step()
        print("loss_gen = %f"%loss_gen)



def make_sample_img(model, z_set, z_size):
    z_set = torch.reshape(z_set, (-1, z_size))
    for id, z in enumerate(z_set):
        z = torch.reshape(z, (1, -1))
        generated_img = model.generator(z)
        generated_img = torch.reshape(generated_img, (28, 28))
        generated_img = generated_img.detach().numpy()

        #plt.imshow(generated_img, cmap='gray', vmin=0., vmax = 1.)
        #plt.show()
        generated_img = ((generated_img-generated_img.min())/(generated_img.max()-generated_img.min())*255.9).astype(np.uint8)
        img = Image.fromarray(generated_img)
        img.save('./sample/img'+str(id)+'.png', 'png')
        #plt.imsave('./sample/img'+str(id)+'.jpeg', generated_img)