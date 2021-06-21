
import torch
import os
from time import time
import matplotlib.pyplot as plt
import numpy as np
import utils

def lr_scheduler(optimizer, epoch, initial_lr = 1e-4, n_epochs=300):
    lr = initial_lr * (1 - (epoch / n_epochs))** 0.9
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        print("learning rate:", lr)

def visualize_label(label, filename, slice_index):
    fig,ax=plt.subplots(1,3,figsize=[16, 9])
    for i in range(0, 3):
        brain_slice = label[0,i,:,:,slice_index]
        ax[i].imshow(brain_slice, cmap="gray")
        ax[i].axis('off')
    plt.plot()
    plt.savefig(filename)

def generate_graphs(train_measures_per_epoch, train_measures_per_iteration, save_path):
    utils.create_graph(save_path=os.path.join(save_path, 'val_loss.png'), x_data=train_measures_per_epoch[:, 0], y_data=train_measures_per_epoch[:, 1])
    utils.create_graph(save_path=os.path.join(save_path, 'val_dice.png'), x_data=train_measures_per_epoch[:, 0], y_data=train_measures_per_epoch[:, 2])
    utils.create_graph(save_path=os.path.join(save_path, 'train_loss.png'), x_data=train_measures_per_epoch[:, 0], y_data=train_measures_per_epoch[:, 3])
    utils.create_graph(save_path=os.path.join(save_path, 'train_dice.png'), x_data=train_measures_per_epoch[:, 0], y_data=train_measures_per_epoch[:, 4])
    utils.create_graph(save_path=os.path.join(save_path, 'iter_loss.png'), x_data=train_measures_per_iteration[:, 2], y_data=train_measures_per_iteration[:, 3])
    utils.create_graph(save_path=os.path.join(save_path, 'iter_dice.png'), x_data=train_measures_per_iteration[:, 2], y_data=train_measures_per_iteration[:, 4])

def valid(valid_dataloader, model, dice_loss, active_contour_loss, focal_loss, dice_coefficient, device, save_folder=None):

    valid_measures=[]
    model.eval()
    total_loss = 0
    dice = 0
    total_iterations = len(valid_dataloader)

    def f(x):
        return 1 if x > 0.5 else 0

    with torch.no_grad():
        val_start = time()
        for i, batch in enumerate(valid_dataloader):

            x_train, y_label = batch['image'], batch['label']
            x_train, y_label = x_train.to(device, dtype=torch.float), y_label.to(device, dtype=torch.float)

            output=model(x_train)
            # print(output.shape)

            d_loss = dice_loss(output, y_label)
            # ac_loss = active_contour_loss(output, y_label)
            # f_loss = focal_loss(output, y_label)
            dice_coeff = dice_coefficient(output, y_label)

            if save_folder:
                output_image = output.clone()
                output_image = output_image.detach().cpu().numpy()
                np.vectorize(f)(output_image)
                y_label_image = y_label.clone()
                y_label_image = y_label_image.detach().cpu().numpy()
                visualize_label(output_image, f'{save_folder}/{i}_output.png', 97)
                visualize_label(y_label_image, f'{save_folder}/{i}_label.png', 97)

            loss = (d_loss)

            total_loss += loss.detach().cpu().numpy()
            dice += dice_coeff.detach().cpu().numpy()

            print(f"Step: Validation, Iteration: {i+1}/{total_iterations} Loss: {loss}, Dice: {dice_coeff}")

            torch.cuda.empty_cache()

    total_loss = total_loss/len(valid_dataloader)
    dice = dice/len(valid_dataloader)
    print(f"Total Loss: {total_loss}, Dice: {dice}, Time taken: {time()-val_start}")

    return total_loss, dice



def train(
    training_dataloader, model, device,
    optimizer, dice_loss, active_contour_loss,
    focal_loss, dice_coefficient, valid_dataloader,
    initial_learning_rate=1e-4, n_epochs=30, save_path=None
):
    # torch.autograd.set_detect_anomaly(True)
    train_measures_per_epoch=[]
    train_measures_per_iteration=[]
    model.train()
    total_iterations = len(training_dataloader)
    best_val_loss = 1000
    model_save_path = os.path.join(save_path, 'best_model.hdf5')

    for epoch in range (1, n_epochs+1):
        epoch_start = time()
        train_loss = 0
        train_dice = 0
        for i, batch in enumerate(training_dataloader):

            model.train()
            iter_start = time()

            x_train, y_label = batch['image'], batch['label']
            x_train, y_label = x_train.to(device, dtype=torch.float), y_label.to(device, dtype=torch.float)
            optimizer.zero_grad()
            output=model(x_train)
           
            d_loss = dice_loss(output, y_label)
            dice_coeff = dice_coefficient(output, y_label)

            loss = d_loss
            loss.backward()
            optimizer.step()

            iter_time = time()-iter_start
            loss_np = loss.detach().cpu().numpy()
            dice_np = dice_coeff.detach().cpu().numpy()

            train_loss += loss_np/len(training_dataloader)
            train_dice += dice_np/len(training_dataloader)

            print(f"Step: Training, Epoch: {epoch}/{n_epochs}, Iteration: {i+1}/{total_iterations} Loss: {loss_np}, Dice: {dice_np}, Time: {iter_time}")

            train_measures_per_iteration.append((epoch, i+1, (epoch-1)*len(training_dataloader)+i, loss_np, dice_np, iter_time))
            torch.cuda.empty_cache()

        lr_scheduler(optimizer, epoch, initial_lr=initial_learning_rate, n_epochs=n_epochs) #do i do epoch+1?

        epoch_end = time()-epoch_start
        print(f"Step: Training, Time taken for epoch {epoch} : {epoch_end}")
        print(f"Average Train Loss: {train_loss}")
        print(f"Average Train Dice: {train_dice}")

        # Run validation
        val_loss, val_dice = 0, 0
        if epoch%5 == 0:
            save_folder = f"epoch_{epoch}"
            if not os.path.exists(save_folder):
                os.makedirs(save_folder, exist_ok=True)
            val_loss, val_dice = valid(valid_dataloader, model, dice_loss, active_contour_loss, focal_loss, dice_coefficient, device=device, save_folder=save_folder)
        else:
            val_loss, val_dice = valid(valid_dataloader, model, dice_loss, active_contour_loss, focal_loss, dice_coefficient, device=device, save_folder=None)

        train_measures_per_epoch.append([epoch, val_loss, val_dice, train_loss, train_dice, epoch_end])

        if save_path:
            generate_graphs(np.array(train_measures_per_epoch), np.array(train_measures_per_iteration), save_path=save_path)
            # Save model if val loss is greater than previous
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"Saving model to {model_save_path} for best loss {val_loss}.")
                torch.save(model.state_dict(), model_save_path)


    return train_measures_per_iteration, train_measures_per_epoch
