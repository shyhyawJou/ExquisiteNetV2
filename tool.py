import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path as p
import time
from os.path import join as pj
import shutil



def train_acc(
        model, save_dir, ds, lf, opt, schdr, 
        total_epoch, device, end_lr, amp
    ):

    max_acc, min_loss = {"train":0, "val":0}, {"train":1000, "val":1000}
    batch_num = len(ds["train"])
    ds_num = len(ds["train"].dataset)
    scaler = GradScaler()
    lr_warmer = LR_Warmer(opt, 40, 1000)
    tb_writer = SummaryWriter(pj(save_dir, "tensorboard"), flush_secs=120)


    t0 = time.time()
    for epoch in range(total_epoch):
        print(f"\nEpoch {epoch+1}/{total_epoch}")
        print('-' * 10)
        tr_correct = 0
        acc, loss = {}, {"train":0}
        # ---------------  start train phase  -------------------------
        model.train()
        for batch_idx, (data, label) in enumerate(ds["train"], 1):
            opt.zero_grad()
            data, label = data.to(device), label.to(device)
            if amp:
                with autocast():
                    output = model(data)
                    batch_loss = lf(output, label)
                scaler.scale(batch_loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                output = model(data)
                batch_loss = lf(output, label)
                batch_loss.backward()
                opt.step()
            lr_warmer.up()
            with torch.no_grad():
                _, pred_label = output.max(1)
                loss["train"] += batch_loss
                tr_correct += torch.sum(pred_label == label)
            print(f"tr set: {batch_idx}/{batch_num}", end='\r')
        # ---------------  end train phase  -------------------------

        acc["train"] = (tr_correct / ds_num)
        loss["train"] = (loss["train"] / batch_num)
        
        min_loss["train"] = min(min_loss["train"], loss["train"])
        max_acc["train"] = max(max_acc["train"], acc["train"])

        print(f"    tr  loss: {loss['train']:.4f}      tr  acc: {acc['train']:.4f}")
        print(f"min tr  loss: {min_loss['train']:.4f}  max tr  acc: {max_acc['train']:.4f}")
        
        loss["val"], acc["val"] = eval_acc(model, ds["val"], lf, device, "val")
        
        if acc["val"] > max_acc["val"]:
            torch.save(model, pj(save_dir, "md.pt"))
            torch.save(model.state_dict(), pj(save_dir, "wt.pt"))

        if acc["val"] == max_acc["val"] and loss["val"] < min_loss["val"]:
            torch.save(model, pj(save_dir, 'md.pt'))
            torch.save(model.state_dict(), pj(save_dir, "wt.pt"))

        min_loss["val"] = min(min_loss["val"], loss["val"])
        max_acc["val"] = max(max_acc["val"], acc["val"])

        print(f"    val loss: {loss['val']:.4f}      val acc: {acc['val']:.4f}")
        print(f"min val loss: {min_loss['val']:.4f}  max val acc: {max_acc['val']:.4f}")

        print(f"LR in this epoch: {opt.param_groups[-1]['lr'] : .6f}")

        schdr.step(loss["train"])

        tb_writer.add_scalars("Loss", loss, epoch + 1)
        tb_writer.add_scalars("Acc", acc, epoch + 1)

        if opt.param_groups[0]['lr'] < end_lr: 
            if lr_warmer.up_count == lr_warmer.steps:
                print('early stop!')
                print("End training")
                break
    
    return time.time()-t0, epoch+1    

def eval_acc(model, ds, lf, device, ds_name):
    model.eval()
    loss, correct = 0, 0
    batch_num = len(ds)
    ds_num = len(ds.dataset)
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(ds, 1):
            data, label = data.to(device), label.to(device)
            outputs = model(data)
            _, pred = outputs.max(1)
            loss += lf(outputs, label)
            correct += torch.sum(pred == label)
            if ds_name is not None:
                print(f"{ds_name} set: {batch_idx}/{batch_num}", end='\r')
        acc = correct / ds_num
        loss /= batch_num
    return loss, acc

def del_ipynb_ckps(root):
    for i in p(root).glob("**/*"):
        if i.name == ".ipynb_checkpoints":
            print(i)
            shutil.rmtree(i)

def create_save_dir(save_root):
    if not p(save_root).exists():
        p(save_root).mkdir(exist_ok=True, parents=True)    

    n = []
    for exp_dir in p(save_root).iterdir():
        if exp_dir.is_dir():
            exp_name = exp_dir.name
            i = -1
            while exp_name[i].isdigit():
                i -= 1
            i += 1
            if i != 0:
                n.append(int(exp_name[i:]))
    
    if len(n) == 0:
        save_dir = pj(save_root, "exp1")
    else:
        save_dir = f"{save_root}/exp{sorted(n)[-1] + 1}"
    
    return save_dir

class LR_Warmer:
    def __init__(self, opt, step=0, scale=1000):    
        assert isinstance(step, int), "step should be int"
        assert scale > 0, "scale should > 0"
        
        if step != 0:
            init_lr, step_size = [], []
            
            for param_group in opt.param_groups:
                lr = param_group["lr"]
                param_group["lr"] /= scale
                init_lr.append(lr)
                step_size.append((lr - lr / scale) / step)
            
            self.opt = opt
            self.step = step
            self.init_lr = init_lr
            self.step_size = step_size
            self.up_count = 0
    
    def up(self):
        if self.up_count != self.step:
            for i, param_group in enumerate(self.opt.param_groups):
                param_group['lr'] += self.step_size[i]
                
            self.up_count += 1
            
            if self.up_count == self.step:
                for i, param_group in enumerate(self.opt.param_groups):
                    param_group['lr'] = self.init_lr[i]




