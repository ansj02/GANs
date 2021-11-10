import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, img_size, z_size):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.z_size = z_size

        self.gen_model = nn.Sequential(
            nn.Linear(z_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, self.img_size[0] * self.img_size[1]),
            nn.Tanh(),
        )

    def forward(self, z):
        generated_img = self.gen_model(z)
        return generated_img


class Discriminator(nn.Module):
    def __init__(self, img_size, drop_prob):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.drop_prob = drop_prob

        self.dis_model = nn.Sequential(
            nn.Linear(self.img_size[0] * self.img_size[1], 128),
            nn.LeakyReLU(),
            nn.Dropout(p=self.drop_prob),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        discriminant_value = self.dis_model(img)
        return discriminant_value


class Model(nn.Module):
    def __init__(self, img_size, z_size, batch_size, drop_prob):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.generator = Generator(img_size, z_size)
        self.discriminator = Discriminator(img_size, drop_prob)
        self.loss = nn.BCELoss()

    def loss_gen(self, z):
        generated_img = self.generator(z)
        discriminant_value = self.discriminator(generated_img)
        loss = torch.sum(torch.log(torch.ones(discriminant_value.shape) - discriminant_value)) / self.batch_size
        return loss

    def loss_gen2(self, z):
        generated_img = self.generator(z)
        discriminant_value = self.discriminator(generated_img)
        loss = self.loss(discriminant_value, torch.ones_like(discriminant_value))
        return loss

    def loss_dis(self, img, z):
        discriminant_value = self.discriminator(img)
        v_gen = self.loss_gen(z)
        loss = -1 * (torch.sum(torch.log(discriminant_value)) / self.batch_size + v_gen)
        return loss

    def loss_dis2(self, img, z):
        real_discriminant_value = self.discriminator(img)
        generated_img = self.generator(z)
        fake_discriminant_value = self.discriminator(generated_img)
        loss_real = self.loss(real_discriminant_value, torch.ones_like(real_discriminant_value))
        loss_fake = self.loss(fake_discriminant_value, torch.zeros_like(fake_discriminant_value))
        loss = (loss_fake + loss_real) / 2
        return loss



