import torch
import numpy as np
from loss import p_losses



def get_print():
    try:
        from logging import get_logger
        logger = get_logger()
        print = logger.info
        return print
    except ImportError:
        return print


def train(
    model,
    dataloader,
    forward_process,
    device,
    optimizer,
    cfg,
    ema=None,
    extra={"skip_steps": 10, "prefix": None},
    logger=None
):
    if ema: model=ema.model
    model.train()
    losses = []
    for step, batch in enumerate(dataloader):

        optimizer.zero_grad()

        batch_size = batch["image"].shape[0]
        batch_imgs = batch["image"].to(device)
        batch_msks = batch["mask"].to(device)

        t = torch.randint(
            1, forward_process.forward_schedule.timesteps, (batch_size,), device=device
        ).long()
        loss, losses_dict = p_losses(
            forward_process,
            model,
            x_start=batch_msks,
            g=batch_imgs,
            t=t,
            cfg=cfg
        )
        losses.append((loss.item(), losses_dict, batch_size))

        loss.backward()
        optimizer.step()
        if ema: ema.update()

        if "skip_steps" in extra.keys():
            if step % extra["skip_steps"] == 0:
                tr_x_total = np.sum([l[-1] for l in losses])

                # ---------- tr losses -------------
                tr_losses_dict = dict()
                for tr_loss in losses:
                    for ln, v in tr_loss[1].items():
                        try:
                            tr_losses_dict[ln] += v * tr_loss[-1]
                        except:
                            tr_losses_dict[ln] = v * tr_loss[-1]
                for k, v in tr_losses_dict.items():
                    tr_losses_dict[k] /= tr_x_total

                extra_tr_losses_txt = ", ".join(
                    [f"{ln}: {v:0.6f}" for ln, v in tr_losses_dict.items()]
                )

                # loss_avg = np.sum([l[0] for l in losses]) / tr_x_total
                prefix = extra.get("prefix", None)
                txt_items = ([prefix,] if prefix else [])
                txt_items.append(f"step:{step:03d}/{len(dataloader)}")
                txt_items.append(
                    f"tr-losses > {extra_tr_losses_txt}"
                )
                
                if logger:
                    logger.info(", ".join(txt_items))
                else:
                    print(", ".join(txt_items))

    return losses, model


@torch.no_grad()
def validate(
    model,
    dataloader,
    forward_process,
    device,
    cfg,
    vl_runs=3,
    logger=None
):
    
    losses = []
    model.eval()
    for step, batch in enumerate(dataloader):

        batch_size = batch["image"].shape[0]
        batch_imgs = batch["image"].to(device)
        batch_msks = batch["mask"].to(device)
        
        _vl_losses = []
        for _ in range(vl_runs):
            t = torch.randint(
                1, forward_process.forward_schedule.timesteps, (batch_size,), device=device
            ).long()
            loss, losses_dict = p_losses(
                forward_process,
                model,
                x_start=batch_msks,
                g=batch_imgs,
                t=t,
                cfg=cfg
            )
            _vl_losses.append((loss.item(), losses_dict))
        
        _vl_avg_loss = np.mean([l[0] for l in _vl_losses])
        _vl_avg_losses_dict = {}
        for k in _vl_losses[0][1].keys():        
            v = np.mean([l[1][k] for l in _vl_losses])
            _vl_avg_losses_dict[k]=v
    
        losses.append((_vl_avg_loss, _vl_avg_losses_dict, batch_size))

    return losses








from metrics import get_binary_metrics
from tqdm import tqdm
from reverse.reverse_process import sample
from utils.helper_funcs import (
    mean_of_list_of_tensors,
)

@torch.no_grad()
def test(
        model,
        dataloader,
        forward_process,
        device,
        cfg,
        vl_runs=3,
        logger=None
):
    losses = []
    model.eval()


    test_metrics = get_binary_metrics()
    ensemble = 1 #5

    for step, batch in tqdm(
        enumerate(dataloader),
        total=len(dataloader),
    ):
        batch_imgs = batch["image"].to(device)
        batch_msks = batch["mask"].to(device)

        samples_list, mid_samples_list = [], []
        for en in range(ensemble):
            samples = sample(
                forward_process.forward_schedule,
                model,
                images=batch_imgs,
                out_channels=batch_msks.shape[1],
                desc=f"ensemble {en+1}/{ensemble}",
            )
            samples_list.append(samples[-1][:, :1, :, :].to(device))

        preds = mean_of_list_of_tensors(samples_list)

        if batch_msks.shape[1] > 1:
            batch_msks = batch_msks[:, 0, :, :].unsqueeze(1)

        if batch_msks.shape[1] > 1:
            preds_ = torch.argmax(preds, 1, keepdim=False).float()
            msks_ = torch.argmax(batch_msks, 1, keepdim=False)
        else:
            preds_ = torch.where(preds > 0, 1, 0).float()
            msks_ = torch.where(batch_msks > 0, 1, 0)
        test_metrics.update(preds_, msks_)

    result = test_metrics.compute()
    return result
