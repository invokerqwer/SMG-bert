import os

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import get_pretrain_dataloader
from utils import get_logger, calculate_recovery, get_angle, get_torsion


class BCEFocalLosswithLogits(nn.Module):
    def __init__(self, gamma=0.2, alpha=0.6, reduction='mean'):
        super(BCEFocalLosswithLogits, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: [N, H, W], target: [N, H, W]
        logits = torch.sigmoid(logits)
        alpha = self.alpha
        gamma = self.gamma
        loss = - alpha * (1 - logits) ** gamma * target * torch.log(logits) - \
               (1 - alpha) * logits ** gamma * (1 - target) * torch.log(1 - logits)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

class Pretrain_Trainer():
    def __init__(self, config, model):
        self.pos_info = config.bert.pos_info
        self.config = config
        self.device = config.device
        self.model = model.to(self.config.device)
        res_dir = self.config.res_dir
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        base_name = os.path.basename(self.config.dataset.smiles_data_path).split(".")[0]

        self.config.log_path = os.path.join(res_dir,base_name+".log")
        self.config.res_path = os.path.join(res_dir,base_name+".csv")

        self.logger = get_logger(config.log_path)

        self.logger.info(f"model_save_dir:{self.config.res_dir},log_path:{self.config.log_path},res_path:{self.config.res_path}, config{config}")

        self.focal_loss = BCEFocalLosswithLogits()

        self.writer = SummaryWriter(os.path.join(res_dir,"./tensorboard"))

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.train.AdamW.lr,
                                           eps=self.config.train.AdamW.eps,
                                           betas=(self.config.train.AdamW.beta1, self.config.train.AdamW.beta2))


        self.train_dataloader, self.valid_dataloader, self.test_dataloader = get_pretrain_dataloader(config,self.logger,smiles_data_path=config.dataset.smiles_data_path,dist_data_path=config.dataset.dist_data_path,
                                                                                                     angle_data_path=config.dataset.angle_data_path,
                                                                                                     torsion_data_path=config.dataset.torsion_data_path)
        self.train_losses_atom_1D = []
        self.train_losses_atom_2D = []
        self.train_losses_atom_3D = []
        self.train_losses_adj = []
        self.train_losses_dist = []
        self.train_losses_angle = []
        self.train_losses_torsion = []
        self.train_losses = []
        self.train_recovery_1D = []
        self.train_recovery_2D = []
        self.train_recovery_3D = []

        self.valid_losses_atom_1D = []
        self.valid_losses_atom_2D = []
        self.valid_losses_atom_3D = []
        self.valid_losses_adj = []
        self.valid_losses_dist = []
        self.valid_losses_angle = []
        self.valid_losses_torsion = []
        self.valid_losses = []
        self.valid_recovery_1D = []
        self.valid_recovery_2D = []
        self.valid_recovery_3D = []

        self.test_losses_atom_1D = []
        self.test_losses_atom_2D = []
        self.test_losses_atom_3D = []
        self.test_losses_adj = []
        self.test_losses_dist = []
        self.test_losses_angle = []
        self.test_losses_torsion = []
        self.test_losses = []
        self.test_recovery_1D = []
        self.test_recovery_2D = []
        self.test_recovery_3D = []


        self.theta_atom_1D = torch.tensor(1., requires_grad=True)
        self.theta_atom_2D = torch.tensor(1., requires_grad=True)
        self.theta_atom_3D = torch.tensor(1., requires_grad=True)
        self.theta_adj = torch.tensor(1., requires_grad=True)
        self.theta_dist = torch.tensor(1., requires_grad=True)
        self.theta_angle = torch.tensor(1., requires_grad=True)
        self.theta_torsion = torch.tensor(1., requires_grad=True)


        self.train_step = 0
        self.val_step = 0

    def train(self, epochs=50):
        train_metric_list = []
        valid_metric_list = []
        for epoch in range(epochs):
            train_metric = self.train_iterations()
            valid_metric = self.valid_test_iterations()
            self.logger.info(f"epoch:{epoch},train_metric:{train_metric},valid_metric:{valid_metric}")
            train_metric_list.append(train_metric)
            valid_metric_list.append(valid_metric)
            self.save_model(os.path.join(self.config.res_dir,str(epoch)+".pth"))
            state = {
                "epoch":epoch,
                "model_state_dict":self.model.state_dict,
                "optimizer":self.optimizer.state_dict(),
                "factor":{
                    "theta_atom_1D":self.theta_atom_1D,
                    "theta_atom_2D":self.theta_atom_2D,
                    "theta_atom_3D":self.theta_atom_3D,
                    "theta_adj":self.theta_adj,
                    "theta_dist": self.theta_dist,
                    "theta_angle": self.theta_angle,
                    "theta_torsion": self.theta_torsion,
                },
                "train_loss":[self.train_losses_atom_1D,
                            self.train_losses_atom_2D,
                            self.train_losses_atom_3D,
                            self.train_losses_adj,
                            self.train_losses_dist,
                            self.train_losses_angle,
                            self.train_losses_torsion,
                            self.train_losses,
                            self.train_recovery_1D,
                            self.train_recovery_2D,
                            self.train_recovery_3D,
                ],
                "val_loss":[
                            self.valid_losses_atom_1D,
                            self.valid_losses_atom_2D,
                            self.valid_losses_atom_3D,
                            self.valid_losses_adj,
                            self.valid_losses_dist,
                            self.valid_losses_angle,
                            self.valid_losses_torsion,
                            self.valid_losses,
                            self.valid_recovery_1D,
                            self.valid_recovery_2D,
                            self.valid_recovery_3D,
                ],
                "test_loss":[
                            self.test_losses_atom_1D,
                            self.test_losses_atom_2D,
                            self.test_losses_atom_3D,
                            self.test_losses_adj,
                            self.test_losses_dist,
                            self.test_losses_angle,
                            self.test_losses_torsion,
                            self.test_losses,
                            self.test_recovery_1D,
                            self.test_recovery_2D,
                            self.test_recovery_3D,
                ]
            }
            torch.save(state, os.path.join(self.config.res_dir,"checkpoint.pth"))


    def train_iterations(self):
        self.model.train()
        losses_atom_1D = []
        losses_atom_2D = []
        losses_atom_3D = []
        losses_adj = []
        losses_dist = []
        losses_angle = []
        losses_torsion = []
        losses = []
        recovery_1D = []
        recovery_2D = []
        recovery_3D = []

        for tra_step, batch_data in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader)):

            self.train_step += 1
            atom_id_raw = batch_data['atom_id_raw'].to(self.device)
            atom_id_mask = batch_data['atom_id_mask'].to(self.device)
            adjoin_matrix = batch_data['adjoin_matrix'].to(self.device)
            dist_matrix = batch_data["dist_matrix"].to(self.device)
            dist_i = batch_data["dist_i"].to(self.device)
            dist_j = batch_data["dist_j"].to(self.device)
            masked_flag = batch_data["masked_flag"].to(self.device)
            gnn_adj = batch_data["gnn_adj"].to(self.device)
            angle = batch_data["angle"].to(self.device)
            angle_i = batch_data["angle_i"].to(self.device)
            angle_j = batch_data["angle_j"].to(self.device)
            angle_k = batch_data["angle_k"].to(self.device)
            torsion_k = batch_data["torsion_k"].to(self.device)
            torsion_i = batch_data["torsion_i"].to(self.device)
            torsion_j = batch_data["torsion_j"].to(self.device)
            torsion_t = batch_data["torsion_t"].to(self.device)
            torsion = batch_data["torsion"].to(self.device)
            dist_mask = batch_data["dist_mask"].to(self.device)

            atom_pred_1D,atom_pred_2D,adj_pred,pos_pred,atom_pred_3D,share,x_atom,x_adj,x_pos = self.model(atom_id_mask,dist_i,dist_j,mask=None,gnn_adj=gnn_adj,dist_score=dist_matrix)
            loss_atom_1D=loss_atom_2D=loss_atom_3D=loss_adj=loss_dist=loss_angle=loss_torsion = torch.tensor(1)

            if self.config.loss.loss_atom_1D:
                loss_atom_1D = Pretrain_Trainer.compute_loss_atom(atom_pred_1D, atom_id_raw, masked_flag)
            if self.config.loss.loss_atom_2D:
                loss_atom_2D = Pretrain_Trainer.compute_loss_atom(atom_pred_2D, atom_id_raw, masked_flag)

            if self.config.loss.adj_loss:
                adj_label = adjoin_matrix.masked_select(dist_mask).unsqueeze(1)
                loss_adj = self.focal_loss(adj_pred, adj_label)
            if self.config.loss.loss_atom_3D:
                loss_atom_3D = Pretrain_Trainer.compute_loss_atom(atom_pred_3D, atom_id_raw, masked_flag)

            if self.config.loss.dist_loss:
                loss_dist = Pretrain_Trainer.compute_loss_dist(dist_matrix, pos_pred, dist_i, dist_j,dist_mask)
            if self.config.loss.angle_loss:
                loss_angle = Pretrain_Trainer.compute_loss_angle(angle, pos_pred, angle_i, angle_j, angle_k)
            if self.config.loss.torsion_loss:
                loss_torsion = Pretrain_Trainer.compute_loss_torsion(torsion, pos_pred, torsion_k, torsion_i, torsion_j, torsion_t)

            if self.config.loss.agg_method == "add":
                loss = loss_atom_1D + loss_atom_2D+loss_atom_3D+loss_adj+loss_dist+loss_angle+loss_torsion
            else:
                loss = 1 / (self.theta_atom_1D ** 2) * loss_atom_1D \
                       + 1 / (self.theta_atom_2D ** 2) * loss_atom_2D \
                       + 1 / (self.theta_atom_3D ** 2) * loss_atom_3D\
                       + 1 /(self.theta_adj ** 2) * loss_adj \
                       + 1 / ( 2 * self.theta_dist ** 2) * loss_dist \
                       + 1 / (2 * self.theta_angle ** 2) * loss_angle + \
                       1 / (2 * self.theta_torsion ** 2) * loss_torsion + \
                       torch.log(self.theta_atom_1D * self.theta_atom_2D*self.theta_atom_3D *self.theta_adj* self.theta_dist * self.theta_angle * self.theta_torsion)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses_atom_1D.append(loss_atom_1D.item())
            losses_atom_2D.append(loss_atom_2D.item())
            losses_atom_3D.append(loss_atom_3D.item())
            losses_adj.append(loss_adj.item())
            losses_dist.append(loss_dist.item())
            losses_angle.append(loss_angle.item())
            losses_torsion.append(loss_torsion.item())
            losses.append(loss.item())

            r_1D = calculate_recovery(atom_pred_1D.cpu(), atom_id_raw.cpu(), masked_flag.cpu())
            recovery_1D.append(r_1D)
            r_2D = calculate_recovery(atom_pred_2D.cpu(), atom_id_raw.cpu(), masked_flag.cpu())
            recovery_2D.append(r_2D)
            r_3D = calculate_recovery(atom_pred_3D.cpu(), atom_id_raw.cpu(), masked_flag.cpu())
            recovery_3D.append(r_3D)


            self.writer.add_scalar('train/losses_atom_1D', loss_atom_1D.item(), self.train_step)
            self.writer.add_scalar('train/losses_atom_2D', loss_atom_2D.item(), self.train_step)
            self.writer.add_scalar('train/losses_atom_3D', loss_atom_3D.item(), self.train_step)
            self.writer.add_scalar('train/losses_adj', loss_adj.item(), self.train_step)
            self.writer.add_scalar('train/losses_dist', loss_dist.item(), self.train_step)
            self.writer.add_scalar('train/losses_angle', loss_angle.item(), self.train_step)
            self.writer.add_scalar('train/losses_torsion', loss_torsion.item(), self.train_step)
            self.writer.add_scalar('train/losses', loss.item(), self.train_step)
            self.writer.add_scalar('train/recovery_1D',r_1D , self.train_step)
            self.writer.add_scalar('train/recovery_2D',r_2D , self.train_step)
            self.writer.add_scalar('train/recovery_3D',r_3D , self.train_step)

        self.train_losses_atom_1D += losses_atom_1D
        self.train_losses_atom_2D += losses_atom_2D
        self.train_losses_atom_3D += losses_atom_3D

        self.train_losses_adj += losses_adj
        self.train_losses_dist += losses_dist
        self.train_losses_angle += losses_angle
        self.train_losses_torsion += losses_torsion
        self.train_losses += losses
        self.train_recovery_1D += recovery_1D
        self.train_recovery_2D += recovery_2D
        self.train_recovery_3D += recovery_3D

        metric = [np.array(loss).mean() for loss in
                  [losses_atom_1D,losses_atom_2D,losses_atom_3D,losses_adj, losses_dist, losses_angle, losses_torsion, losses, recovery_1D,recovery_2D,recovery_3D]]
        names = ["loss_atom_1D","loss_atom_2D","loss_atom_3D","loss_adj", "loss_dist", "loss_angle", "loss_torsion", "loss", "recovery_1D","recovery_2D","recovery_3D"]
        return dict(zip(names, metric))
    @staticmethod
    def compute_loss_dist(dist_labels, pos, dist_i, dist_j,dist_mask):
        dist_labels = dist_labels[:,:dist_mask.shape[1],:dist_mask.shape[1]].masked_select(dist_mask)
        pos = pos.reshape(-1,pos.shape[-1])
        dist_pred = ((pos[dist_i] - pos[dist_j]).pow(2).sum(dim=-1)+1e-9).sqrt()
        loss = torch.nn.functional.mse_loss(dist_pred, dist_labels)
        return loss

    @staticmethod
    def compute_loss_torsion(torsion_labels, pos, torsion_k, torsion_i, torsion_j, torsion_t):
        pos = pos.reshape(-1, pos.shape[-1])
        torsion_pred = get_torsion(pos, torsion_k, torsion_i, torsion_j, torsion_t)
        loss = torch.nn.functional.mse_loss(torsion_pred, torsion_labels)
        return loss

    @staticmethod
    def compute_loss_angle(angle_labels, pos, angle_i, angle_j, angle_k):
        pos = pos.reshape(-1, pos.shape[-1])
        angle_pred = get_angle(pos, angle_i, angle_j, angle_k)
        loss = torch.nn.functional.mse_loss(angle_pred, angle_labels)
        return loss

    @staticmethod
    def compute_loss_atom(atom_pred, atom_labels, masked_flag):
        atom_pred = atom_pred[masked_flag == 1]
        atom_labels = atom_labels[masked_flag == 1]
        loss_atom = torch.nn.functional.cross_entropy(atom_pred, atom_labels)
        return loss_atom

    def save_model(self, model_save_path):
        torch.save(self.model.state_dict(), model_save_path)

    def valid_test_iterations(self, mode='valid'):
        self.model.eval()
        if mode == 'test':
            dataloader = self.test_dataloader
            self_losses = self.test_losses
            self_losses_atom_1D = self.test_losses_atom_1D
            self_losses_atom_2D = self.test_losses_atom_2D
            self_losses_atom_3D = self.test_losses_atom_3D
            self_losses_adj = self.test_losses_adj
            self_losses_dist = self.test_losses_dist
            self_losses_angle = self.test_losses_angle
            self_losses_torsion = self.test_losses_angle
            self_recovery_1D = self.test_recovery_1D
            self_recovery_2D = self.test_recovery_2D
            self_recovery_3D = self.test_recovery_3D
        elif mode == "valid":
            dataloader = self.valid_dataloader
            self_losses = self.valid_losses
            self_losses_atom_1D = self.valid_losses_atom_1D
            self_losses_atom_2D = self.valid_losses_atom_2D
            self_losses_atom_3D = self.valid_losses_atom_3D
            self_losses_adj = self.valid_losses_adj
            self_losses_dist = self.valid_losses_dist
            self_losses_angle = self.valid_losses_angle
            self_losses_torsion = self.valid_losses_angle
            self_recovery_1D = self.valid_recovery_1D
            self_recovery_2D = self.valid_recovery_2D
            self_recovery_3D = self.valid_recovery_3D
        else:
            raise Exception("No such mode")

        with torch.no_grad():
            losses_atom_1D = []
            losses_atom_2D = []
            losses_atom_3D = []
            losses_adj = []
            losses_dist = []
            losses_angle = []
            losses_torsion = []
            losses = []
            recovery_1D = []
            recovery_2D = []
            recovery_3D = []

            for i, batch_data in enumerate(dataloader):
                self.val_step += 1
                atom_id_raw = batch_data['atom_id_raw'].to(self.device)
                atom_id_mask = batch_data['atom_id_mask'].to(self.device)
                adjoin_matrix = batch_data['adjoin_matrix'].to(self.device)
                dist_matrix = batch_data["dist_matrix"].to(self.device)
                dist_i = batch_data["dist_i"].to(self.device)
                dist_j = batch_data["dist_j"].to(self.device)
                masked_flag = batch_data["masked_flag"].to(self.device)
                gnn_adj = batch_data["gnn_adj"].to(self.device)
                angle = batch_data["angle"].to(self.device)
                angle_i = batch_data["angle_i"].to(self.device)
                angle_j = batch_data["angle_j"].to(self.device)
                angle_k = batch_data["angle_k"].to(self.device)
                torsion_k = batch_data["torsion_k"].to(self.device)
                torsion_i = batch_data["torsion_i"].to(self.device)
                torsion_j = batch_data["torsion_j"].to(self.device)
                torsion_t = batch_data["torsion_t"].to(self.device)
                torsion = batch_data["torsion"].to(self.device)
                dist_mask = batch_data["dist_mask"].to(self.device)

                atom_pred_1D,atom_pred_2D,adj_pred,pos_pred,atom_pred_3D,share,x_atom,x_adj,x_pos = self.model(atom_id_mask,dist_i,dist_j,mask=None,gnn_adj=gnn_adj,dist_score=dist_matrix)

                loss_atom_1D = loss_atom_2D = loss_atom_3D = loss_adj = loss_dist = loss_angle = loss_torsion = torch.tensor(
                    1)

                if self.config.loss.loss_atom_1D:
                    loss_atom_1D = Pretrain_Trainer.compute_loss_atom(atom_pred_1D, atom_id_raw, masked_flag)
                if self.config.loss.loss_atom_2D:
                    loss_atom_2D = Pretrain_Trainer.compute_loss_atom(atom_pred_2D, atom_id_raw, masked_flag)

                if self.config.loss.adj_loss:
                    adj_label = adjoin_matrix.masked_select(dist_mask).unsqueeze(1)
                    loss_adj = self.focal_loss(adj_pred, adj_label)
                if self.config.loss.loss_atom_3D:
                    loss_atom_3D = Pretrain_Trainer.compute_loss_atom(atom_pred_3D, atom_id_raw, masked_flag)

                if self.config.loss.dist_loss:
                    loss_dist = Pretrain_Trainer.compute_loss_dist(dist_matrix, pos_pred, dist_i, dist_j, dist_mask)
                if self.config.loss.angle_loss:
                    loss_angle = Pretrain_Trainer.compute_loss_angle(angle, pos_pred, angle_i, angle_j, angle_k)
                if self.config.loss.torsion_loss:
                    loss_torsion = Pretrain_Trainer.compute_loss_torsion(torsion, pos_pred, torsion_k, torsion_i,
                                                                         torsion_j, torsion_t)

                if self.config.loss.agg_method == "add":
                    loss = loss_atom_1D + loss_atom_2D + loss_atom_3D + loss_adj + loss_dist + loss_angle + loss_torsion
                else:
                    loss = 1 / (self.theta_atom_1D ** 2) * loss_atom_1D \
                           + 1 / (self.theta_atom_2D ** 2) * loss_atom_2D \
                           + 1 / (self.theta_atom_3D ** 2) * loss_atom_3D \
                           + 1 / (self.theta_adj ** 2) * loss_adj \
                           + 1 / (2 * self.theta_dist ** 2) * loss_dist \
                           + 1 / (2 * self.theta_angle ** 2) * loss_angle + \
                           1 / (2 * self.theta_torsion ** 2) * loss_torsion + \
                           torch.log(
                               self.theta_atom_1D * self.theta_atom_2D * self.theta_atom_3D * self.theta_adj * self.theta_dist * self.theta_angle * self.theta_torsion)

                losses_atom_1D.append(loss_atom_1D.item())
                losses_atom_2D.append(loss_atom_2D.item())
                losses_atom_3D.append(loss_atom_3D.item())
                losses_adj.append(loss_adj.item())
                losses_dist.append(loss_dist.item())
                losses_angle.append(loss_angle.item())
                losses_torsion.append(loss_torsion.item())
                losses.append(loss.item())

                r_1D = calculate_recovery(atom_pred_1D.cpu(), atom_id_raw.cpu(), masked_flag.cpu())
                recovery_1D.append(r_1D)
                r_2D = calculate_recovery(atom_pred_2D.cpu(), atom_id_raw.cpu(), masked_flag.cpu())
                recovery_2D.append(r_2D)
                r_3D = calculate_recovery(atom_pred_3D.cpu(), atom_id_raw.cpu(), masked_flag.cpu())
                recovery_3D.append(r_3D)

                self.writer.add_scalar('val/losses_atom_1D', loss_atom_1D.item(), self.val_step)
                self.writer.add_scalar('val/losses_atom_2D', loss_atom_2D.item(), self.val_step)
                self.writer.add_scalar('val/losses_atom_3D', loss_atom_3D.item(), self.val_step)
                self.writer.add_scalar('val/losses_adj', loss_adj.item(), self.val_step)
                self.writer.add_scalar('val/losses_dist', loss_dist.item(), self.val_step)
                self.writer.add_scalar('val/losses_angle', loss_angle.item(), self.val_step)
                self.writer.add_scalar('val/losses_torsion', loss_torsion.item(), self.val_step)
                self.writer.add_scalar('val/losses', loss.item(), self.val_step)
                self.writer.add_scalar('val/recovery_1D',r_1D , self.val_step)
                self.writer.add_scalar('val/recovery_2D',r_2D , self.val_step)
                self.writer.add_scalar('val/recovery_3D',r_3D , self.val_step)
            self_losses_atom_1D += losses_atom_1D
            self_losses_atom_2D += losses_atom_2D
            self_losses_atom_3D += losses_atom_3D

            self_losses_adj += losses_adj
            self_losses_dist += losses_dist
            self_losses_angle += losses_angle
            self_losses_torsion += losses_torsion
            self_losses += losses
            self_recovery_1D += recovery_1D
            self_recovery_2D += recovery_2D
            self_recovery_3D += recovery_3D
            metric = [np.array(loss).mean() for loss in
                      [losses_atom_1D,losses_atom_2D,losses_atom_3D,losses_adj, losses_dist, losses_angle, losses_torsion, losses, recovery_1D,recovery_2D,recovery_3D]]
            names = ["loss_atom_1D","loss_atom_2D","loss_atom_3D","loss_adj", "loss_dist", "loss_angle", "loss_torsion", "loss", "recovery_1D","recovery_2D","recovery_3D"]
            return dict(zip(names, metric))