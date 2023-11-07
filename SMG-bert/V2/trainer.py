
import pandas as pd
from easydict import EasyDict


from utils import get_logger, calculate_recovery, get_torsion,get_dist,get_angle
import os
from sklearn.metrics import r2_score, roc_auc_score
import numpy as np
import torch
from tqdm import tqdm
from datasets import get_dataloader
from torch.utils.tensorboard import SummaryWriter
class Pretrain_Trainer_3D():
    def __init__(self, config, model):
        self.nmr = config.bert.nmr
        self.pos_info = config.bert.pos_info

        self.config = config
        self.device = config.device
        self.model = model.to(self.config.device)
        res_dir = self.config.res_dir
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        base_name = os.path.basename(self.config.dataset.dataset_path).split(".")[0]
        self.config.log_path = os.path.join(res_dir,base_name+".log")
        self.config.res_path = os.path.join(res_dir,base_name+".csv")
        self.logger = get_logger(config.log_path)
        self.logger.info(f"model_save_dir:{self.config.res_dir},log_path:{self.config.log_path},res_path:{self.config.res_path}")
        self.logger.info(config)
        self.writer = SummaryWriter(os.path.join(res_dir,"./tensorboard"))



        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.train.AdamW.lr,
                                           eps=self.config.train.AdamW.eps,
                                           betas=(self.config.train.AdamW.beta1, self.config.train.AdamW.beta2))
        self.train_dataloader, self.valid_dataloader, self.test_dataloader = get_dataloader(config, self.logger,pretrain=True,use_3D=True,pin_memory=True)
        self.train_losses_atom = []
        self.train_losses_nmr = []
        self.train_losses_dist = []
        self.train_losses_angle = []
        self.train_losses_torsion = []
        self.train_losses = []
        self.train_recovery = []

        self.valid_losses_atom = []
        self.valid_losses_nmr = []
        self.valid_losses_dist = []
        self.valid_losses_angle = []
        self.valid_losses_torsion = []
        self.valid_losses = []
        self.valid_recovery = []

        self.test_losses_atom = []
        self.test_losses_nmr = []
        self.test_losses_dist = []
        self.test_losses_angle = []
        self.test_losses_torsion = []
        self.test_losses = []
        self.test_recovery = []

        self.theta_atom = torch.tensor(1., requires_grad=True)
        self.theta_nmr = torch.tensor(1., requires_grad=True)
        self.theta_dist = torch.tensor(1., requires_grad=True)
        self.theta_angle = torch.tensor(1., requires_grad=True)
        self.theta_torsion = torch.tensor(1., requires_grad=True)
        self.train_step = 0
        self.val_step = 0

    def train(self, epochs=50):
        train_metric_list = []
        valid_metric_list = []
        for epoch in range(1,epochs+1):


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
                "theta_atom":self.theta_atom,
                "theta_nmr": self.theta_nmr,
                "theta_dist": self.theta_dist,
                "theta_angle": self.theta_angle,
                "theta_torsion": self.theta_torsion,
            }
            torch.save(state, os.path.join(self.config.res_dir,"checkpoint.pth"))


    def train_iterations(self):
        self.model.train()
        losses_atom = []
        losses_nmr = []
        losses_dist = []
        losses_angle = []
        losses_torsion = []
        losses = []
        recovery = []

        for tra_step, batch_data in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader)):
            self.train_step+=1

            adjoin_matrix = batch_data['adjoin_matrix'].to(self.device)
            mask = batch_data['mask'].to(self.device)
            masked_mol_ids = batch_data["masked_mol_ids"].to(self.device)
            masked_nmr = batch_data["masked_nmr"].to(self.device)
            nmr_labels = batch_data["nmr_labels"].to(self.device).float()
            atom_labels = batch_data["atom_labels"].to(self.device)
            masked_flag = batch_data["masked_flag"].to(self.device)

            pos_raw = batch_data["pos"].to(self.device)
            dist_i = batch_data["dist_i"].to(self.device)
            dist_j = batch_data["dist_j"].to(self.device)
            dist = batch_data["dist"].to(self.device)
            angle_i = batch_data["angle_i"].to(self.device)
            angle_j = batch_data["angle_j"].to(self.device)
            angle_k = batch_data["angle_k"].to(self.device)
            angle = batch_data["angle"].to(self.device)
            torsion_k = batch_data["torsion_k"].to(self.device)
            torsion_i = batch_data["torsion_i"].to(self.device)
            torsion_j = batch_data["torsion_j"].to(self.device)
            torsion_t = batch_data["torsion_t"].to(self.device)
            torsion = batch_data["torsion"].to(self.device)

            nmr, atom, pos, x_atom, x_nmr, x_pos, share = self.model(masked_mol_ids, masked_nmr,mask=mask, adjoin_matrix=adjoin_matrix)
            loss_nmr = loss_dist = loss_angle = loss_torsion = torch.tensor(1.)

            loss_atom = Pretrain_Trainer_3D.compute_loss_atom(atom, atom_labels, masked_flag)
            if self.nmr:
                loss_nmr = Pretrain_Trainer_3D.compute_loss_nmr(nmr, nmr_labels, masked_flag)
                self.writer.add_scalar('train/losses_nmr', loss_nmr.item(), self.train_step)
            if self.pos_info:
                loss_dist = Pretrain_Trainer_3D.compute_loss_dist(dist, pos, dist_i, dist_j)
                loss_angle = Pretrain_Trainer_3D.compute_loss_angle(angle, pos, angle_i, angle_j, angle_k)
                loss_torsion = Pretrain_Trainer_3D.compute_loss_torsion(torsion, pos, torsion_k, torsion_i, torsion_j, torsion_t)
                self.writer.add_scalar('train/losses_dist', loss_dist.item(), self.train_step)
                self.writer.add_scalar('train/losses_angle', loss_angle.item(), self.train_step)
                self.writer.add_scalar('train/losses_torsion', loss_torsion.item(), self.train_step)

            loss = 1 / (self.theta_atom ** 2) * loss_atom + 1 / (self.theta_nmr ** 2) * loss_nmr + 1 / (
                        2 * self.theta_dist ** 2) * loss_dist + 1 / (2 * self.theta_angle ** 2) * loss_angle + 1 / (
                               2 * self.theta_torsion ** 2) * loss_torsion + torch.log(
                self.theta_atom * self.theta_nmr * self.theta_dist * self.theta_angle * self.theta_torsion)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses_atom.append(loss_atom.item())
            losses_nmr.append(loss_nmr.item())
            losses_dist.append(loss_dist.item())
            losses_angle.append(loss_angle.item())
            losses_torsion.append(loss_torsion.item())
            losses.append(loss.item())
            r= calculate_recovery(atom.cpu(), atom_labels.cpu(), masked_flag.cpu())
            recovery.append(r)
            self.writer.add_scalar('train/losses_atom', loss_atom.item(), self.train_step)

            self.writer.add_scalar('train/losses', loss.item(), self.train_step)
            self.writer.add_scalar('train/recovery',r , self.train_step)

        self.train_losses_atom += losses_atom
        self.train_losses_nmr += losses_nmr
        self.train_losses_dist += losses_dist
        self.train_losses_angle += losses_angle
        self.train_losses_torsion += losses_torsion
        self.train_losses += losses
        self.train_recovery += recovery
        metric = [np.array(loss).mean() for loss in
                  [losses_atom, losses_nmr, losses_dist, losses_angle, losses_torsion, losses, recovery]]
        names = ["loss_atom", "loss_nmr", "loss_dist", "loss_angle", "loss_torsion", "loss", "recovery"]
        return dict(zip(names, metric))

    def compute_loss_dist(dist_labels, pos, dist_i, dist_j):
        pos = pos.reshape(-1, pos.shape[-1])
        dist_pred = get_dist(pos, dist_i, dist_j)
        loss = torch.nn.functional.mse_loss(dist_pred, dist_labels)
        return loss

    def compute_loss_torsion(torsion_labels, pos, torsion_k, torsion_i, torsion_j, torsion_t):
        pos = pos.reshape(-1, pos.shape[-1])
        torsion_pred = get_torsion(pos, torsion_k, torsion_i, torsion_j, torsion_t)
        loss = torch.nn.functional.mse_loss(torsion_pred, torsion_labels)
        return loss

    def compute_loss_angle(angle_labels, pos, angle_i, angle_j, angle_k):
        pos = pos.reshape(-1, pos.shape[-1])
        angle_pred = get_angle(pos, angle_i, angle_j, angle_k)
        loss = torch.nn.functional.mse_loss(angle_pred, angle_labels)
        return loss

    def compute_loss_atom(atom_pred, atom_labels, masked_flag):
        atom_pred = atom_pred[masked_flag == 1]

        atom_labels = atom_labels[masked_flag == 1]
        loss_atom = torch.nn.functional.cross_entropy(atom_pred, atom_labels)
        return loss_atom

    def compute_loss_nmr( nmr_pred, nmr_labels, masked_flag):
        nmr_pred = nmr_pred[masked_flag == 1].squeeze()
        nmr_labels = nmr_labels[masked_flag == 1]
        loss_nmr = torch.nn.functional.mse_loss(nmr_pred, nmr_labels)
        return loss_nmr

    def save_model(self, model_save_path):
        torch.save(self.model.state_dict(), model_save_path)

    def valid_test_iterations(self, mode='valid'):
        self.model.eval()
        if mode == 'test':
            dataloader = self.test_dataloader
            self_losses = self.test_losses
            self_losses_atom = self.test_losses_atom
            self_losses_nmr = self.test_losses_nmr
            self_losses_dist = self.test_losses_dist
            self_losses_angle = self.test_losses_angle
            self_losses_torsion = self.test_losses_angle
            self_recovery = self.test_recovery
        elif mode == "valid":
            dataloader = self.valid_dataloader
            self_losses = self.valid_losses
            self_losses_atom = self.valid_losses_atom
            self_losses_nmr = self.valid_losses_nmr
            self_recovery = self.valid_recovery
            self_losses_dist = self.valid_losses_dist
            self_losses_angle = self.valid_losses_angle
            self_losses_torsion = self.valid_losses_angle
        else:
            raise Exception("No such mode")

        with torch.no_grad():

            losses_atom = []
            losses_nmr = []
            losses_dist = []
            losses_angle = []
            losses_torsion = []
            losses = []
            recovery = []
            for i, batch_data in enumerate(dataloader):
                self.val_step += 1

                adjoin_matrix = batch_data['adjoin_matrix'].to(self.device)
                mask = batch_data['mask'].to(self.device)
                masked_mol_ids = batch_data["masked_mol_ids"].to(self.device)
                masked_nmr = batch_data["masked_nmr"].to(self.device)
                nmr_labels = batch_data["nmr_labels"].to(self.device).float()
                atom_labels = batch_data["atom_labels"].to(self.device)
                masked_flag = batch_data["masked_flag"].to(self.device)

                pos_raw = batch_data["pos"].to(self.device)
                dist_i = batch_data["dist_i"].to(self.device)
                dist_j = batch_data["dist_j"].to(self.device)
                dist = batch_data["dist"].to(self.device)
                angle_i = batch_data["angle_i"].to(self.device)
                angle_j = batch_data["angle_j"].to(self.device)
                angle_k = batch_data["angle_k"].to(self.device)
                angle = batch_data["angle"].to(self.device)
                torsion_k = batch_data["torsion_k"].to(self.device)
                torsion_i = batch_data["torsion_i"].to(self.device)
                torsion_j = batch_data["torsion_j"].to(self.device)
                torsion_t = batch_data["torsion_t"].to(self.device)
                torsion = batch_data["torsion"].to(self.device)

                nmr, atom, pos, x_atom, x_nmr, x_pos, share = self.model(masked_mol_ids, masked_nmr,mask=mask, adjoin_matrix=adjoin_matrix)
                loss_nmr = loss_dist = loss_angle = loss_torsion = torch.tensor(1.)

                loss_atom = Pretrain_Trainer_3D.compute_loss_atom(atom, atom_labels, masked_flag)
                if self.nmr:
                    loss_nmr = Pretrain_Trainer_3D.compute_loss_nmr(nmr, nmr_labels, masked_flag)
                    self.writer.add_scalar('val/losses_nmr', loss_nmr.item(), self.val_step)
                if self.pos_info:
                    loss_dist = Pretrain_Trainer_3D.compute_loss_dist(dist, pos, dist_i, dist_j)
                    loss_angle = Pretrain_Trainer_3D.compute_loss_angle(angle, pos, angle_i, angle_j, angle_k)
                    loss_torsion = Pretrain_Trainer_3D.compute_loss_torsion(torsion, pos, torsion_k, torsion_i, torsion_j, torsion_t)
                    self.writer.add_scalar('val/losses_dist', loss_dist.item(), self.val_step)
                    self.writer.add_scalar('val/losses_angle', loss_angle.item(), self.val_step)
                    self.writer.add_scalar('val/losses_torsion', loss_torsion.item(), self.val_step)

                loss = 1 / (self.theta_atom ** 2) * loss_atom + 1 / (self.theta_nmr ** 2) * loss_nmr + 1 / (
                        2 * self.theta_dist ** 2) * loss_dist + 1 / (2 * self.theta_angle ** 2) * loss_angle + 1 / (
                               2 * self.theta_torsion ** 2) * loss_torsion + torch.log(
                    self.theta_atom * self.theta_nmr * self.theta_dist * self.theta_angle * self.theta_torsion)

                losses_atom.append(loss_atom.item())
                losses_nmr.append(loss_nmr.item())
                losses_dist.append(loss_dist.item())
                losses_angle.append(loss_angle.item())
                losses_torsion.append(loss_torsion.item())
                losses.append(loss.item())
                r = calculate_recovery(atom.cpu(), atom_labels.cpu(), masked_flag.cpu())
                recovery.append(r)
                self.writer.add_scalar('val/losses_atom', loss_atom.item(), self.val_step)
                self.writer.add_scalar('val/losses', loss.item(), self.val_step)
                self.writer.add_scalar('val/recovery', r, self.val_step)
            self_losses_atom += losses_atom
            self_losses_nmr += losses_nmr
            self_losses_dist += losses_dist
            self_losses_angle += losses_angle
            self_losses_torsion += losses_torsion
            self_losses += losses
            self_recovery += recovery
            metric = [np.array(loss).mean() for loss in
                      [losses_atom, losses_nmr, losses_dist, losses_angle, losses_torsion, losses, recovery]]
            names = ["loss_atom", "loss_nmr", "loss_dist", "loss_angle", "loss_torsion", "loss", "recovery"]
            return dict(zip(names, metric))