import torch
import numpy as np
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def _take_channels(*xs, ignore_channels=None):
    if ignore_channels is None:
        return xs
    else:
        channels = [
            channel
            for channel in range(xs[0].shape[1])
            if channel not in ignore_channels
        ]
        xs = [
            torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device))
            for x in xs
        ]
        return xs


def iou(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate Intersection over Union between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    intersection = torch.sum(gt * pr)
    union = torch.sum(gt) + torch.sum(pr) - intersection + eps
    return (intersection + eps) / union


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def plot_image_mask(img, mask, alpha=0.5):
    img = (img * 255).to(torch.uint8)
    obj_ids = torch.unique(mask)
    obj_ids = obj_ids[1:]
    mask = mask == obj_ids[:, None, None]

    show(draw_segmentation_masks(img, masks=mask, alpha=alpha, colors="blue"))


def evaluate_model(model, val_dataloader, threshold=0.5, device="cpu"):
    model.eval()
    model.to(device)
    iou_scores = []
    for input, mask in tqdm(val_dataloader):
        input = input.to(device)
        mask = mask.to(device)

        with torch.no_grad():
            logits = model(input)
        pr_masks = logits

        input = input.cpu().detach()
        mask = mask.cpu().detach()
        pr_masks = pr_masks.cpu().detach()

        for image, gt_mask, pr_mask in zip(input, mask, pr_masks):
            iou_scores.append(iou(pr_mask, gt_mask, threshold=threshold))

    return np.mean(iou_scores)


def train_epoch(model, loss_fn, optimizer, train_dataloader, device="cpu"):
    model.train()
    model.to(device)

    losses = []
    for input, mask in tqdm(train_dataloader):
        input = input.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        prediction = model.forward(input)
        loss = loss_fn(prediction, mask)
        loss.backward()
        optimizer.step()

        loss_value = loss.cpu().detach().numpy()
        losses.append(loss_value)

    return np.mean(losses)


def val_epoch(model, loss_fn, val_dataloader, device="cpu"):
    model.eval()
    model.to(device)

    losses = []
    for input, mask in tqdm(val_dataloader):
        input = input.to(device)
        mask = mask.to(device)

        with torch.no_grad():
            prediction = model.forward(input)
            loss = loss_fn(prediction, mask)
        loss_value = loss.cpu().detach().numpy()
        losses.append(loss_value)

    return np.mean(losses)


def train(
    model,
    train_dataloader,
    val_dataloader,
    loss_fn,
    optimizer,
    scheduler,
    device="cpu",
    num_epochs=5,
):
    for epoch in range(num_epochs):
        train_loss = train_epoch(
            model, loss_fn, optimizer, train_dataloader, device=device
        )
        val_loss = val_epoch(model, loss_fn, val_dataloader, device=device)
        iou_score = evaluate_model(model, val_dataloader, threshold=0.5, device=device)

        print(f"Epoch: {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}")
        print(f"IOU score: {iou_score}")
