from Utils.AvgMeter import AvgMeter
from tqdm import tqdm
from Utils.mr2mr import *

def valid_epoch(model, valid_loader, accuracies_req, verbose=True):
    # print tqdm progress bar if verbose=True, else only print the result after each epoch.
    loss_meter = AvgMeter(len(accuracies_req))

    if verbose:
        iter_object = tqdm(valid_loader, total=len(valid_loader))
    else:
        iter_object = valid_loader

    for batch in iter_object:
        loss, mr1_embeddings, mr2_embeddings = model(batch)

        mr1_mr2_acc = mr2mr_match(mr1_embeddings, mr2_embeddings, accuracies_req)
        mr2_mr1_acc = mr2mr_match(mr2_embeddings, mr1_embeddings, accuracies_req)

        loss_meter.update(loss.item(), mr1_mr2_acc, mr2_mr1_acc)
        if verbose:
            iter_object.set_postfix(valid_loss=loss_meter.avg, mr1_to_mr2_accuracy=loss_meter.mr1_to_mr2_accuracy,
                                    mr2_to_mr1_accuracy=loss_meter.mr2_to_mr1_accuracy)

    print(
        f"[Valid] loss: {loss_meter.avg}, mr1_to_mr2_accuracy: {loss_meter.mr1_to_mr2_accuracy}, mr2_to_mr1_accuracy: {loss_meter.mr2_to_mr1_accuracy}")

    return loss_meter
