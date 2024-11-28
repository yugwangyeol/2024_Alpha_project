import os
import argparse
import torch
import time
import pickle
import numpy as np

from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import VideoAnomalyDataset_C3D, VideoAnomalyDataset_C3D_for_Clip
from models import model

from tqdm import tqdm
from aggregate import remake_video_output, evaluate_auc

torch.backends.cudnn.benchmark = False

# Config
def get_configs():
    parser = argparse.ArgumentParser(description="VAD-Jigsaw config")
    parser.add_argument("--val_step", type=int, default=5000) # 검증 주기
    parser.add_argument("--print_interval", type=int, default=100) # 학습 중 로그를 출력할 간격
    parser.add_argument("--epochs", type=int, default=100) # 전체 학습 반복 수
    parser.add_argument("--gpu_id", type=str, default=0) # GPU ID를 지정하여 사용할 GPU를 설정
    parser.add_argument("--log_date", type=str, default=None) # 로그를 저장할 날짜 및 시간
    parser.add_argument("--batch_size", type=int, default=192) # 학습 시 배치 크기
    parser.add_argument("--static_threshold", type=float, default=0.2) # 정적 프레임을 판단하는 임계값
    parser.add_argument("--sample_num", type=int, default=7) # 한 비디오에서 사용할 프레임의 개수
    parser.add_argument("--clip_num", type=int, default=5)
    parser.add_argument("--checkpoint", type=str, default=None) # 
    parser.add_argument("--dataset", type=str, default="DAD_Jigsaw")
    parser.add_argument("--data_type", type=str, default='top_IR', 
                        choices=['front_depth', 'front_IR', 'top_depth', 'top_IR'])
    parser.add_argument("--save_epoch", type=int, default=1)
    args = parser.parse_args()

    args.device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")
    return args


def train(args):
    if not args.log_date:
        running_date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    else:
        running_date = args.log_date
    print("The running_data : {}".format(running_date))
    for k,v in vars(args).items():
        print("-------------{} : {}".format(k, v))

    # Load Data
    data_dir = f"/home/work/Alpha/Jigsaw-VAD/{args.dataset}/training/{args.data_type}/frames"

    vad_dataset = VideoAnomalyDataset_C3D(data_dir, 
                                          frame_num=args.sample_num,
                                          static_threshold=args.static_threshold)
    #vad_clip_dataset = VideoAnomalyDataset_C3D_clip(data_dir,clip_num=args.clip_num, static_threshold=None) #############################################

    vad_dataloader = DataLoader(vad_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    net = model.WideBranchNet(time_length=args.sample_num, num_classes=[args.sample_num ** 2, 81])
    # ime_length는 입력 프레임 수를, num_classes는 분류할 클래스 수를 나타냄
    # args.sample_num ** 2는 sample_num 값에 따라 달라짐

    if args.checkpoint is not None:
        state = torch.load(args.checkpoint)
        print('load ' + args.checkpoint)
        net.load_state_dict(state, strict=True)
        net.cuda()
        smoothed_auc, smoothed_auc_avg, _ = val(args, net)
        exit(0)
    # 만약 checkpoint가 설정되어 있다면, 해당 체크포인트 파일에서 모델 가중치를 로드하고, val 함수에서 검증을 수행한 후 종료

    net.cuda(args.device)
    net = net.train()

    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.Adam(params=net.parameters(), lr=1e-4)

    # Train
    log_dir = './log/{}/'.format(running_date)
    writer = SummaryWriter(log_dir)
    # SummaryWriter를 사용하여 텐서보드 로그를 저장할 writer 객체를 생성함 -> 로그 디렉토리는 log_dir로 설정

    t0 = time.time()
    global_step = 0

    max_acc = -1
    timestamp_in_max = None
    
    frame_loss_sum = 0  # 프레임 로스를 누적할 변수
    stacked_clips = []  # 클립을 저장할 리스트

    for epoch in range(args.epochs):
        for it, data in enumerate(vad_dataloader):
            # 데이터 로드
            video, clip, temp_labels, spat_labels, t_flag, clip_org, phase = \
                data['video'], data['clip'], data['label'], data["trans_label"], data["temporal"], data["clip_org"], data["phase"]

            clip = clip.cuda(args.device, non_blocking=True)
            temp_labels = temp_labels[t_flag].long().view(-1).cuda(args.device)
            spat_labels = spat_labels[~t_flag].long().view(-1).cuda(args.device)

            # 프레임 로스 계산
            temp_logits, spat_logits = net(clip)
            temp_logits = temp_logits[t_flag].view(-1, args.sample_num)
            spat_logits = spat_logits[~t_flag].view(-1, 9)

            temp_loss = criterion(temp_logits, temp_labels)
            spat_loss = criterion(spat_logits, spat_labels)

            frame_loss = temp_loss + spat_loss
            frame_loss_sum += frame_loss  # 프레임 로스 누적

            # 클립 데이터 저장
            stacked_clips.append(clip_org)

            # 5개의 프레임 로스가 누적되면 클립 로스를 계산
            if len(stacked_clips) == args.clip_num:
                vad_dataset_clip = VideoAnomalyDataset_C3D_for_Clip(
                    clips=torch.stack(stacked_clips),
                    frame_num=args.sample_num,
                    static_threshold=args.static_threshold,
                    phase=phase
                )
                vad_dataloader_clip = DataLoader(
                    vad_dataset_clip, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True
                )

                # 클립 로스 계산
                for clip_data in vad_dataloader_clip:
                    batch_clips, clip_temp_labels, clip_spat_labels = \
                        clip_data["clips"], clip_data["labels"], clip_data["trans_label"]

                    clip_temp_logits, clip_spat_logits = net(batch_clips)
                    clip_temp_loss = criterion(clip_temp_logits, clip_temp_labels)
                    clip_spat_loss = criterion(clip_spat_logits, clip_spat_labels)

                    clip_loss = clip_temp_loss + clip_spat_loss

                    # 최종 손실 계산 및 업데이트
                    total_loss = frame_loss_sum / args.clip_num + clip_loss  # 평균 프레임 로스와 클립 로스 합산
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                    # Tensorboard에 로스 기록
                    writer.add_scalar('Train/frame_loss', (frame_loss_sum / args.clip_num).item(), global_step=global_step)
                    writer.add_scalar('Train/clip_loss', clip_loss.item(), global_step=global_step)
                    writer.add_scalar('Train/total_loss', total_loss.item(), global_step=global_step)

                # 스택 및 프레임 로스 초기화
                stacked_clips = []
                frame_loss_sum = 0

            global_step += 1

                
            ####################################
            
            
            global_step += 1

            if global_step % args.val_step == 0 and epoch >= 5: # 수정
                smoothed_auc, smoothed_auc_avg, temp_timestamp = val(args, net)
                writer.add_scalar('Test/smoothed_auc', smoothed_auc, global_step=global_step)
                writer.add_scalar('Test/smoothed_auc_avg', smoothed_auc_avg, global_step=global_step)

                if smoothed_auc > max_acc:
                    max_acc = smoothed_auc
                    timestamp_in_max = temp_timestamp
                    save = './checkpoint/{}_{}.pth'.format('best', running_date)
                    torch.save(net.state_dict(), save)

                print('cur max: ' + str(max_acc) + ' in ' + timestamp_in_max)
                net = net.train()
        
        if epoch % args.save_epoch == 0 and epoch >= 5:
            save = './save_ckpt/{}_{}.pth'.format(running_date,epoch)
            torch.save(net.state_dict(), save)
            
def compare_clips(clip1, clip2):
    diff = (clip1 - clip2).abs().sum().item()
    print(f"Difference between clips: {diff}")

def val(args, net=None):
    if not args.log_date:
        running_date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    else:
        running_date = args.log_date
    print("The running_date : {}".format(running_date))

    # Load Data
    data_dir = f"/home/work/Alpha/Jigsaw-VAD/{args.dataset}/testing/{args.data_type}/frames" #

    testing_dataset = VideoAnomalyDataset_C3D(data_dir, 
                                              frame_num=args.sample_num)
    testing_data_loader = DataLoader(testing_dataset, batch_size=256, shuffle=False, num_workers=4, drop_last=False)

    net.eval()

    video_output = {}
    for idx, data in enumerate(tqdm(testing_data_loader)):
        videos = data["video"]
        frames = data["frame"].tolist()
        clip = data["clip"].cuda(args.device)

        with torch.no_grad():
            temp_logits, spat_logits = net(clip)
            #print("Temp logits1:", temp_logits)
            if torch.isnan(temp_logits).any(): # 수정
                print(f"NaN detected in output at batch {idx}")
                print(f"Input statistics: min={data['clip'].min()}, max={data['clip'].max()}")

            temp_logits = temp_logits.view(-1, args.sample_num, args.sample_num)
            #print("Temp logits2:", temp_logits)

            spat_logits = spat_logits.view(-1, 9, 9)

        spat_probs = F.softmax(spat_logits, -1)
        diag = torch.diagonal(spat_probs, offset=0, dim1=-2, dim2=-1)
        scores = diag.min(-1)[0].cpu().numpy()
        # spat_logits 값을 소프트맥스 함수(softmax)를 사용하여 확률로 변환하고, 주 대각선 요소를 선택하여 각 로짓의 최소값을 scores에 저장

        temp_probs = F.softmax(temp_logits, -1)
        diag2 = torch.diagonal(temp_probs, offset=0, dim1=-2, dim2=-1)
        #print(diag2.shape)
        scores2 = diag2.min(-1)[0].cpu().numpy()
        # temp_logits 값을 소프트맥스 함수(softmax)를 사용하여 확률로 변환하고, 주 대각선 요소를 선택하여 각 로짓의 최소값을 scores에 저장
        
        for video_, frame_, s_score_, t_score_  in zip(videos, frames, scores, scores2):
            #print(s_score_ ,' ' ,t_score_)
            if video_ not in video_output:
                video_output[video_] = {}
            if frame_ not in video_output[video_]:
                video_output[video_][frame_] = []
            video_output[video_][frame_].append([s_score_, t_score_])
            #print(s_score_,t_score_)
        # video_output 딕셔너리에 video_, frame_을 키로 하고, s_score_, t_score_ 값을 저장하여 각 비디오와 프레임의 성능을 기록

    micro_auc, macro_auc = save_and_evaluate(video_output, running_date, dataset=args.dataset)
    return micro_auc, macro_auc, running_date


def save_and_evaluate(video_output, running_date, dataset='DAD_Jigsaw'):
    pickle_path = './log/video_output_ori_{}.pkl'.format(running_date)
    with open(pickle_path, 'wb') as write:
        pickle.dump(video_output, write, pickle.HIGHEST_PROTOCOL)
    video_output_spatial, video_output_temporal, video_output_complete = remake_video_output(video_output, dataset=dataset)
    evaluate_auc(video_output_spatial, dataset=dataset)
    evaluate_auc(video_output_temporal, dataset=dataset)
    smoothed_res, smoothed_auc_list = evaluate_auc(video_output_complete, dataset=dataset)
    return smoothed_res.auc, np.mean(smoothed_auc_list)
#

if __name__ == '__main__':
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')
    args = get_configs()
    train(args)