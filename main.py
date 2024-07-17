import torch
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import random
import numpy as np
from src.models.evflownet import EVFlowNet
from src.datasets import DatasetProvider
from enum import Enum, auto
from src.datasets import train_collate
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any
import os
import time
import torch.nn as nn
import torch.nn.functional as F






class RepresentationType(Enum):
    VOXEL = auto()
    STEPAN = auto()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def compute_epe_error(pred_flow: torch.Tensor, gt_flow: torch.Tensor):
    '''
    end-point-error (ground truthと予測値の二乗誤差)を計算
    pred_flow: torch.Tensor, Shape: torch.Size([B, 2, 480, 640]) => 予測したオプティカルフローデータ
    gt_flow: torch.Tensor, Shape: torch.Size([B, 2, 480, 640]) => 正解のオプティカルフローデータ
    '''
    #epe = torch.mean(torch.mean(torch.norm(pred_flow - gt_flow, p=2, dim=1), dim=(1, 2)), dim=0)
    gt_flow_resized = nn.functional.interpolate(gt_flow, size=(pred_flow.size(2), pred_flow.size(3)), mode='bilinear', align_corners=False)
    epe = torch.mean(torch.norm(pred_flow - gt_flow_resized, p=2, dim=1))
    return epe

def save_optical_flow_to_npy(flow: torch.Tensor, file_name: str):
    '''
    optical flowをnpyファイルに保存
    flow: torch.Tensor, Shape: torch.Size([2, 480, 640]) => オプティカルフローデータ
    file_name: str => ファイル名
    '''
    np.save(f"{file_name}.npy", flow.cpu().numpy())


####################################################################
# multi_scale_loss関数の追加
def multi_scale_loss(predictions, target):
    losses = []
    for prediction in predictions:
        loss = compute_epe_error(prediction, target)
        losses.append(loss)
    return sum(losses)

import os
import torch

def find_latest_checkpoint(checkpoint_dir):
    checkpoint_files = [file for file in os.listdir(checkpoint_dir) if file.endswith('.pth')]
    if not checkpoint_files:
        return None
    latest_file = max(checkpoint_files, key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x)))
    return os.path.join(checkpoint_dir, latest_file)

def load_checkpoint(model, optimizer, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']

####################################################################

@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(args: DictConfig):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    '''
        ディレクトリ構造:

        data
        ├─test
        |  ├─test_city
        |  |    ├─events_left
        |  |    |   ├─events.h5
        |  |    |   └─rectify_map.h5
        |  |    └─forward_timestamps.txt
        └─train
            ├─zurich_city_11_a
            |    ├─events_left
            |    |       ├─ events.h5
            |    |       └─ rectify_map.h5
            |    ├─ flow_forward
            |    |       ├─ 000134.png
            |    |       |.....
            |    └─ forward_timestamps.txt
            ├─zurich_city_11_b
            └─zurich_city_11_c
        '''
    
    # ------------------
    #    Dataloader
    # ------------------
    loader = DatasetProvider(
        dataset_path=Path(args.dataset_path),
        representation_type=RepresentationType.VOXEL,
        delta_t_ms=100,
        num_bins=4
    )
    train_set = loader.get_train_dataset()
    test_set = loader.get_test_dataset()
    collate_fn = train_collate
    train_data = DataLoader(train_set,
                                 batch_size=args.data_loader.train.batch_size,
                                 shuffle=args.data_loader.train.shuffle,
                                 collate_fn=collate_fn,
                                 drop_last=False)
    test_data = DataLoader(test_set,
                                 batch_size=args.data_loader.test.batch_size,
                                 shuffle=args.data_loader.test.shuffle,
                                 collate_fn=collate_fn,
                                 drop_last=False)

    '''
    train data:
        Type of batch: Dict
        Key: seq_name, Type: list
        Key: event_volume, Type: torch.Tensor, Shape: torch.Size([Batch, 4, 480, 640]) => イベントデータのバッチ
        Key: flow_gt, Type: torch.Tensor, Shape: torch.Size([Batch, 2, 480, 640]) => オプティカルフローデータのバッチ
        Key: flow_gt_valid_mask, Type: torch.Tensor, Shape: torch.Size([Batch, 1, 480, 640]) => オプティカルフローデータのvalid. ベースラインでは使わない
    
    test data:
        Type of batch: Dict
        Key: seq_name, Type: list
        Key: event_volume, Type: torch.Tensor, Shape: torch.Size([Batch, 4, 480, 640]) => イベントデータのバッチ
    '''
    # ------------------
    #       Model
    # ------------------

    ################################################################################################################
    #model_path = None
    ################################################################################################################

    model = EVFlowNet(args.train).to(device)

    # ------------------
    #   optimizer
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.train.initial_learning_rate, weight_decay=args.train.weight_decay)
    # ------------------
    #   Start training
    # ------------------
    model.train()

    """checkpoint_dir = './checkpoints'  # チェックポイントのディレクトリを指定
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        start_epoch, last_loss = load_checkpoint(model, optimizer, latest_checkpoint)
        start_epoch += 1  # 最後に保存されたエポックの次から開始
        print(f"Resuming training from epoch {start_epoch} with loss {last_loss}")"""
    
    ####################################################################
    #for epoch in range(start_epoch, args.train.epochs):
    ####################################################################
    for epoch in range(args.train.epochs):
        total_loss = 0
        print("on epoch: {}".format(epoch+1))
        for i, batch in enumerate(tqdm(train_data)):
            batch: Dict[str, Any]
            #event_image = batch["event_volume"].to(device) # [B, 4, 480, 640]
            #ground_truth_flow = batch["flow_gt"].to(device) # [B, 2, 480, 640]
            #flow = model(event_image) # [B, 2, 480, 640]
            #loss: torch.Tensor = compute_epe_error(flow, ground_truth_flow)
            ####################################################################
            
            event_image_pair = batch["event_volume"]  # [2, B, 4, 480, 640]
            ground_truth_flow = batch["flow_gt"].to(device)  # [B, 2, 480, 640]
            ground_truth_flow = ground_truth_flow.squeeze(1)

            #print(f'event_image_pair[0]: {event_image_pair[0].size()}')
            #print(f'event_image_pair[1]: {event_image_pair[1].size()}')
            #print(f"ground_truth_flow: {ground_truth_flow.size()}")

            event_image_pair = [img.to(device) for img in event_image_pair]  # 各フレームをデバイスに移動

            # モデルに渡すために連結
            event_images = torch.cat(event_image_pair, dim=1)  # [B, 8, 480, 640] (4チャネル * 2フレーム)
            
          

            flow_predictions = model(event_images)
            #flow_predictions = model(event_image)
            loss = multi_scale_loss(flow_predictions, ground_truth_flow)
            ####################################################################
            #modelが多重スケールの出力を返すようにEVFlowNetを変更する
            
            
            print(f"batch {i} loss: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_data)}')
        current_time = time.strftime("%Y%m%d%H%M%S")
        model_path = f"checkpoints/model_epoch_{epoch + 1}_{current_time}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
        }, model_path)
        #files.download(model_path)
        print(f"Model checkpoint saved to {model_path}")

    # Create the directory if it doesn't exist
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    
    current_time = time.strftime("%Y%m%d%H%M%S")
    model_path = f"checkpoints/model_{current_time}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # ------------------
    #   Start predicting
    # ------------------
    ################################################################################################################
    #if model_path is None:
      #model_path = "/content/dl_lecture_competition_pub/checkpoints/model_20240714133025.pth"
    ################################################################################################################
    model.load_state_dict(torch.load(model_path, map_location=device))
    #print('model is gained')
    model.eval()
    flow: torch.Tensor = torch.tensor([]).to(device)
    with torch.no_grad():
        print("start test")
        for batch in tqdm(test_data):
            batch: Dict[str, Any]
            #event_image = batch["event_volume"].to(device)
            #batch_flow = model(event_image) # [1, 2, 480, 640]
            #flow = torch.cat((flow, batch_flow), dim=0)  # [N, 2, 480, 640]
            ################################################################################################################
            event_image_pair = batch["event_volume"]  # [2, B, 4, 480, 640]
            event_images = [img.to(device) for img in event_image_pair]  # 各フレームをデバイスに移動
            event_images = torch.cat(event_images, dim=1)  # チャンネル次元で連結 [B, 8, 480, 640] torch.Size([1, 8, 480, 640]
            #print(event_images.shape)#torch.Size([1, 8, 480, 640]
            batch_flow = model(event_images)[-1]  # モデルに連結したイメージを入力
            """for idx, flow in enumerate(batch_flow):
              print(f"Shape of output {idx}: {flow.size()}")"""
            flow = torch.cat((flow, batch_flow), dim=0)  # 結果を連結して全体のフローを構築       
            ################################################################################################################
        print(flow.shape)
    
        print("test done")
    # ------------------
    #  save submission
    # ------------------
    file_name = "submission"
    save_optical_flow_to_npy(flow, file_name)

if __name__ == "__main__":
    main()

