import torch

from Modules import Generator, Discriminator
from Datasets import UnpairedData
from torch import optim, save, load
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from train_one_step import train_one_step
import time
import os


gen_params = {"layer_dims": [3, 8, 16, 32, 64, 128, 256],
            "data_size": 128,
            "bottle_neck_size": 32,
            "latent_size": 32,
            "layer_depth": 5}


disc_params = {"layer_dims": [3, 8, 16, 32, 64, 128, 256],
            "data_size": 128,
            "output_size": 1,
            "layer_depth": 2}


optim_params = {"lr": 0.002,
                   "betas": (0.9, 0.999),
                   "eps": 1e-08,
                   "weight_decay": 0.1}

training_params = {"batch_size": 128,
                   "training_epochs": 2000,
                   "data1_path": "../Data/boobs/images_crop",
                   "data2_path": "../Data/boobs/images_crop",
                   "cuda": True,
                   "Sampling_rate": 30,
                   "Save_rate": 500,
                   "Save_path": './models',
                   "Load_model": False,
                   "Load_Path": ""}

# create moodel
GEN1 = Generator(**gen_params)
GEN2 = Generator(**gen_params)
DISC1 = Discriminator(**disc_params)
DISC2 = Discriminator(**disc_params)

def save_models(save_path, id):

    folder_path = os.path.join(save_path, id)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    save(GEN1, folder_path + '/GEN1')
    save(GEN2, folder_path + '/GEN2')
    save(DISC1, folder_path + '/DISC1')
    save(DISC2, folder_path + '/DISC2')


if training_params["Load_model"]:

    GEN1 = load(training_params["Load_Path"] + "/GEN1")
    GEN2 = load(training_params["Load_Path"] + "/GEN2")
    DISC1 = load(training_params["Load_Path"] + "/DISC1")
    DISC2 = load(training_params["Load_Path"] + "/DISC2")


if training_params["cuda"]:
    print('Model: Cuda')
    GEN1.to('cuda')
    GEN2.to('cuda')
    DISC1.to('cuda')
    DISC1.to('cuda')


# define optimizer
optimizerG1 = optim.Adam(GEN1.parameters(), **optim_params)
optimizerG2 = optim.Adam(GEN2.parameters(), **optim_params)
optimizerD1 = optim.Adam(DISC1.parameters(), **optim_params)
optimizerD2 = optim.Adam(DISC2.parameters(), **optim_params)

# define Criterions

criterion1 = torch.nn.BCELoss()
criterion2 = torch.nn.L1Loss()


# create training set
trainingSet = UnpairedData(training_params["data1_path"], training_params["data2_path"])


# Create data loader
data_loader = DataLoader(trainingSet, batch_size=training_params["batch_size"],
                        shuffle=True, num_workers=4, drop_last=True)

# Writer will output to ./runs/ directory by default
writer = SummaryWriter()


# Training Loop TODO
print(f'Starting training for {training_params["training_epochs"]} epochs')

for epoch in range(training_params["training_epochs"]):
    start_time = time.time()
    print(f'\nEpoch {epoch} of {training_params["training_epochs"]}:')


    for counter, batch in enumerate(data_loader):
        batch_time = time.time()

        input_A = batch['image_A'].float()
        input_B = batch['image_B'].float()
        Latent = torch.rand(training_params["batch_size"], gen_params["latent_size"])

        if training_params["cuda"]:
            input_A = input_A.cuda()
            input_B = input_B.cuda()

        FakeImageA, FakeImageB, G1Loss, D1Loss, G2Loss, D2Loss = train_one_step(GEN1, GEN2, DISC1, DISC2,
                                                                                input_A, input_B, Latent,
                                                                                optimizerG1, optimizerG2,
                                                                                optimizerD1, optimizerD2,
                                                                                criterion1, criterion2)



        step_count = (counter + epoch*training_params["batch_size"])
        if step_count % training_params["Sampling_rate"] == 0:
            print(f'\rBatch Time: {(time.time() - batch_time):.2f} seconds', end='')
            images_per_batch = 4
            image_grid = make_grid(torch.cat((input_B[:images_per_batch],
                                             FakeImageB[:images_per_batch],
                                             input_A[:images_per_batch],
                                             FakeImageA[:images_per_batch]), 0))

            writer.add_image('Images', image_grid, step_count)
            writer.add_scalar('G1Loss',  G1Loss , step_count)
            writer.add_scalar('D1Loss', D1Loss, step_count)
            writer.add_scalar('G2Loss', G2Loss, step_count)
            writer.add_scalar('D2Loss', D2Loss, step_count)

    print(f' Epoch Time: {(time.time() - start_time):.2f} seconds')

    if epoch%training_params["Save_rate"] == 0:
        print("Saving...")
        save_models(training_params["Save_path"], id)
        print("Done")


writer.close()




