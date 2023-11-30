import torch

def calculate_top_accuracy(predicted_labels, k):
    correct_count = 0
    total_count = len(predicted_labels)

    for i in range(total_count):
        prediction = predicted_labels[i]

        if i in prediction[:k]:
            correct_count += 1

    top_accuracy = correct_count / total_count * 100
    return top_accuracy


def mr2mr_match(mr1_embeddings, mr2_embeddings, accuracies_req):
    scores_matrix = torch.matmul(mr1_embeddings, mr2_embeddings.T)
    top_indices = torch.argsort(scores_matrix, dim=1, descending=True)[:, :]
    predicted_labels = top_indices.tolist()

    top_accuracy_list = []
    for k in accuracies_req:
        accuracy = calculate_top_accuracy(predicted_labels, k)
        top_accuracy_list.append(accuracy)

    accuracy_tensor = torch.tensor(top_accuracy_list, device='cpu')
    return accuracy_tensor






