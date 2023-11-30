from Utils.AvgMeter import AvgMeter
from tqdm import tqdm
from Utils.mr2mr import *
def train_epoch(model, train_loader, optimizer, lr_scheduler, step, accuracies_req, verbose=True):
    # print tqdm progress bar if verbose=True, else only print the result after each epoch.
    loss_meter = AvgMeter(len(accuracies_req))

    if verbose:
        iter_object = tqdm(train_loader, total=len(train_loader))
    else:
        iter_object = train_loader

    for batch in iter_object:
        loss, mr1_embeddings, mr2_embeddings = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        # do not allow gradient in calculating accuracy.
        mr1_mr2_acc = mr2mr_match(mr1_embeddings.detach(), mr2_embeddings.detach(), accuracies_req)
        mr2_mr1_acc = mr2mr_match(mr2_embeddings.detach(), mr1_embeddings.detach(), accuracies_req)

        loss_meter.update(loss.item(), mr1_mr2_acc, mr2_mr1_acc)
        loss_meter.get_lr(optimizer)
        if verbose:
            iter_object.set_postfix(train_loss=loss_meter.avg, lr=loss_meter.lr, mr1_to_mr2_accuracy=loss_meter.mr1_to_mr2_accuracy,
                                    mr2_to_mr1_accuracy=loss_meter.mr2_to_mr1_accuracy)

    print(
        f"[Train] loss: {loss_meter.avg}, mr1_to_mr2_accuracy: {loss_meter.mr1_to_mr2_accuracy}, mr2_to_mr1_accuracy: {loss_meter.mr2_to_mr1_accuracy}")

    return loss_meter
