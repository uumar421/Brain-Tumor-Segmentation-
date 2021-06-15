import torch
import torch.nn as nn
from monai.metrics.meandice import compute_meandice
from monai.metrics.utils import do_metric_reduction

class soft_dice_loss(nn.Module):
    def __init__(self, epsilon=1e-8, reduce_axis=None):
        super().__init__()
        self.epsilon=epsilon
        if not reduce_axis:
            self.reduce_axis = [2, 3, 4]
        else:
            self.reduce_axis = reduce_axis

    def forward(self, y_pred, y_true):

        numerator = 2 * torch.sum(y_true * y_pred, dim=self.reduce_axis) + self.epsilon
        denominator = torch.sum(torch.pow(y_true, 2), dim=self.reduce_axis) + torch.sum(torch.pow(y_pred, 2), dim=self.reduce_axis) + self.epsilon
        dice_loss = 1 - torch.div(numerator, denominator)

        return torch.mean(dice_loss)


class active_contour_loss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon=epsilon

    def forward(self, y_pred, y_true):
        #volumetric_term
        y_pred_clone = y_pred.clone()
        y_true_clone = y_true.clone()

        c1 = torch.ones_like(y_pred_clone)
        c2 = torch.zeros_like(y_pred_clone)

        vol_loss = torch.mean(y_pred_clone*torch.pow(c1 - y_true_clone, 2)) + torch.mean((1 - y_pred_clone)*(torch.pow(c2 - y_true_clone, 2)))
        #length_term
        y_pred_clone[y_pred_clone >= 0.5] = 1
        y_pred_clone[y_pred_clone < 0.5] = 0
        # print(y_true_clone.shape)
        # print(y_pred_clone.shape)

        delta_x = y_pred_clone[:,:, 1:, :, :] - y_pred_clone[:,:, :-1, :, :]  # x gradient (B, H-1, W, D)
        delta_y = y_pred_clone[:,:, :, 1:, :] - y_pred_clone[:,:, :, :-1, :]  # y gradient (B, H, W-1, D)
        delta_z = y_pred_clone[:,:, :, :, 1:] - y_pred_clone[:,:, :, :, :-1]  # z gradient (B, H, W, D-1)

        delta_x = delta_x[:, :, 1:, :-2, :-2]
        delta_y = delta_y[:, :, :-2, 1:, :-2]
        delta_z = delta_z[:, :, :-2, :-2, 1:]

        len_loss = torch.mean(torch.sqrt(torch.abs(torch.pow(delta_x, 2) + torch.pow(delta_y, 2) + torch.pow(delta_z, 2)) + self.epsilon))
        ac_loss = vol_loss + len_loss

        return ac_loss


class focal_loss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon=epsilon

    def forward(self, y_pred, y_true):

        focal_loss = -torch.mean(torch.pow(1-y_pred, 2) * y_true * torch.log(y_pred + self.epsilon))

        return focal_loss


class dice_coefficient(nn.Module):
    def __init__(self, epsilon=1e-8, reduce_axis=None):
        super().__init__()
        self.epsilon=epsilon
        if not reduce_axis:
            self.reduce_axis = [2, 3, 4]
        else:
            self.reduce_axis = reduce_axis

    def forward(self, y_pred, y_true):

        dice_numerator = 2 * torch.sum(y_true * y_pred, dim=self.reduce_axis) + self.epsilon
        dice_denominator = torch.sum(y_true, dim=self.reduce_axis) + torch.sum(y_pred, dim=self.reduce_axis) + self.epsilon
        dice_coefficient = torch.mean((dice_numerator)/(dice_denominator))

        return torch.mean(dice_coefficient)

class multi_class_dice_coefficient(nn.Module):
    def __init__(self, epsilon=1e-8, reduce_axis=None):
        super().__init__()
        self.epsilon=epsilon
        if not reduce_axis:
            self.reduce_axis = [2, 3, 4]
        else:
            self.reduce_axis = reduce_axis

    def forward(self, y_pred, y_true):

        dice_numerator = 2 * torch.sum(y_true * y_pred, dim=self.reduce_axis) + self.epsilon
        dice_denominator = torch.sum(y_true, dim=self.reduce_axis) + torch.sum(y_pred, dim=self.reduce_axis) + self.epsilon
        dice_coefficient = (dice_numerator)/(dice_denominator)

        return dice_coefficient

def sens_spec(pred, label, class_num, threshold=0.5):
    sensitivities = []
    specificities = []
    for i in range(class_num):
        class_pred = pred[:,i, :, :, :]
        class_pred = torch.where(class_pred>threshold, 1, 0)
        class_label = label[:, i, :, :, :]

        tp = torch.sum(torch.logical_and((class_pred == 1),(class_label == 1)))

        tn = torch.sum(torch.logical_and((class_pred == 0) , (class_label == 0)))
        
        fp = torch.sum(torch.logical_and((class_pred == 1) , (class_label == 0)))
        
        fn = torch.sum(torch.logical_and((class_pred == 0) , (class_label == 1)))

        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        sensitivity = torch.nan_to_num(sensitivity, nan=1.0)

        print(f"Class: {i}, TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}, Sens: {sensitivity}, Spec: {specificity}")

        sensitivities.append(float(sensitivity.detach().cpu().numpy()))
        specificities.append(float(specificity.detach().cpu().numpy()))
    
    return sensitivities, specificities


class dice_coefficient_monai(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):

        dice_coefficient = compute_meandice(y_pred, y_true)

        return do_metric_reduction(dice_coefficient)[0]
