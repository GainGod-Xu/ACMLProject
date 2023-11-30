import torch
class AvgMeter:
    def __init__(self, accuracies_req_num=7):
        self.name = "Metric"
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
        self.lr = 0.0
        self.mr1_to_mr2_accuracy = torch.zeros(accuracies_req_num)
        self.mr2_to_mr1_accuracy = torch.zeros(accuracies_req_num)
        self.mr1_to_mr2_accuracy_sum = torch.zeros(accuracies_req_num)
        self.mr2_to_mr1_accuracy_sum = torch.zeros(accuracies_req_num)

    def update(self, loss, mr1_2_accuracy, mr2_1_accuracy):
        self.count += 1
        self.sum += loss
        self.avg = self.sum / self.count
        self.mr1_to_mr2_accuracy_sum += mr1_2_accuracy
        self.mr2_to_mr1_accuracy_sum += mr2_1_accuracy
        self.mr1_to_mr2_accuracy = self.mr1_to_mr2_accuracy_sum/self.count
        self.mr2_to_mr1_accuracy = self.mr2_to_mr1_accuracy_sum/self.count

    def get_lr(self, optimizer):
        self.lr = optimizer.param_groups[0]['lr']
        return self.lr

    def __repr__(self):
        text = f"{self.name}: {self.avg:.8f}, MR1 to MR2 Accuracy: {self.mr1_to_mr2_accuracy}, MR2 to MR1 Accuracy: {self.mr2_to_mr1_accuracy}"
        return text
