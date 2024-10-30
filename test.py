from dataset import ETRIDataset_color
from networks import *
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torch.utils.data.distributed
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    """ The main function of the test process for performance measurement. """
    net = efficientformer_color().to(DEVICE)
    trained_weights = torch.load('./model/model_11.pt',map_location=DEVICE)
    net.load_state_dict(trained_weights)
    net.eval()
    
    # 아래 경로는 포함된 샘플(validation set)의 경로로, 실제 추론환경에서의 경로는 task.ipynb를 참고 바랍니다.
    df = pd.read_csv('../Dataset/Fashion-How24_sub2_val.csv')
    val_dataset = ETRIDataset_color(df, base_path='../Dataset/val/', mode='val')
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=10)
    
    gt_list = np.array([])
    pred_list = np.array([])
    features_list = []
    
    for j, sample in enumerate(tqdm(val_dataloader, leave=False)):
        for key in sample:
            sample[key] = sample[key].to(DEVICE)
            
        out = net(sample)
        gt = np.array(sample['color_label'].cpu())
        gt_list = np.concatenate([gt_list, gt], axis=0)
        _, indx = out.max(1)
        pred_list = np.concatenate([pred_list, indx.cpu()], axis=0)
        features_list.append(out.cpu().detach().numpy())
        
    features = np.concatenate(features_list, axis=0)
    tsne = TSNE(n_components=2, random_state=214)
    tsne_results = tsne.fit_transform(features)
    
    top_1, acsa = get_test_metrics(gt_list, pred_list)
    print("------------------------------------------------------")
    print(
        "Color: Top-1=%.5f, ACSA=%.5f" % (top_1, acsa))
    print("------------------------------------------------------")
    
    # plt.figure(figsize=(16, 10))
    # scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=gt_list, cmap='tab20')
    # plt.colorbar(scatter)
    
    # t-SNE 시각화와 범례 추가
    # plt.figure(figsize=(16, 10))
    # cmap = plt.get_cmap('tab20', np.unique(gt_list).size)
    # scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=gt_list, cmap=cmap)

    # # 레이블을 가진 컬러 바 추가
    # cbar = plt.colorbar(scatter, ticks=np.arange(np.unique(gt_list).size))
    # cbar.set_label('Class Labels')
    # cbar.set_ticks(np.arange(np.unique(gt_list).size))
    # cbar.set_ticklabels(np.unique(gt_list))
    
    # plt.title("t-SNE visualization of features")
    # plt.xlabel("t-SNE component 1")
    # plt.ylabel("t-SNE component 2")
    # plt.show()
    
    return top_1

def get_test_metrics(y_true, y_pred, verbose=True):
    y_true, y_pred = y_true.astype(np.int8), y_pred.astype(np.int8)
    cnf_matrix = confusion_matrix(y_true, y_pred)
    
    if verbose:
        print(cnf_matrix)
        
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    
    top_1 = np.sum(TP)/np.sum(np.sum(cnf_matrix))
    cs_accuracy = TP / cnf_matrix.sum(axis=1)
    
    return top_1, cs_accuracy.mean()

if __name__ == '__main__':
    main()