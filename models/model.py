import torch
import torch.nn as nn
import torch.nn.functional as F

class WideBranchNet(nn.Module):
    def __init__(self, frame_num=7, clip_num=5):
        super(WideBranchNet, self).__init__()
        
        self.frame_num = frame_num
        self.clip_num = clip_num
        
        # Backbone network
        self.backbone = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            
            nn.Conv3d(32, 64, kernel_size=(5, 3, 3), padding=(2, 1, 1), bias=False),
            nn.InstanceNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(5, 3, 3), padding=(2, 1, 1), bias=False),
            nn.InstanceNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(128),
            nn.ReLU(),
        )
        
        # 모드별 시간적 특징 처리기
        self.frame_temporal_processor = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=(5, 3, 3), padding=(2, 1, 1)),
            nn.InstanceNorm3d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool3d((7, 4, 4))  # frame mode용
        )
        
        self.clip_temporal_processor = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=(5, 3, 3), padding=(2, 1, 1)),
            nn.InstanceNorm3d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool3d((35, 4, 4))  # clip mode용
        )
        
        # 공간적 특징 처리기
        self.spatial_processor = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.InstanceNorm3d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool3d((1, 4, 4))
        )
        
        # Feature sizes
        frame_temporal_feat_size = 128 * 7 * 4 * 4
        clip_temporal_feat_size = 128 * 35 * 4 * 4
        spatial_feat_size = 128 * 1 * 4 * 4
        
        # 분류기
        self.frame_temp_classifier = self._make_classifier(frame_temporal_feat_size, frame_num**2)
        self.frame_spat_classifier = self._make_classifier(spatial_feat_size, 81)
        self.clip_temp_classifier = self._make_classifier(clip_temporal_feat_size, clip_num**2)
        self.clip_spat_classifier = self._make_classifier(spatial_feat_size, 81)
        
    def _make_classifier(self, in_features, num_classes):
        return nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x, mode='frame'):
        batch_size = x.size(0)
        
        # 공통 특징 추출
        features = self.backbone(x)
        
        if mode == 'frame':
            temp_features = self.frame_temporal_processor(features)
            temp_features = temp_features.view(batch_size, -1)
            temp_out = self.frame_temp_classifier(temp_features)
            temp_out = temp_out.view(batch_size, -1, self.frame_num)  # [B, frame_num, frame_num]
            
            spat_features = self.spatial_processor(features)
            spat_features = spat_features.view(batch_size, -1)
            spat_out = self.frame_spat_classifier(spat_features)
            spat_out = spat_out.view(batch_size, -1, 9)  # [B, 9, 9]
            
        else:  # clip mode
            temp_features = self.clip_temporal_processor(features)
            temp_features = temp_features.view(batch_size, -1)
            temp_out = self.clip_temp_classifier(temp_features)
            temp_out = temp_out.view(batch_size, -1, self.clip_num)  # [B, clip_num, clip_num]
            
            spat_features = self.spatial_processor(features)
            spat_features = spat_features.view(batch_size, -1)
            spat_out = self.clip_spat_classifier(spat_features)
            spat_out = spat_out.view(batch_size, -1, 9)  # [B, 9, 9]
        
        return temp_out, spat_out

if __name__ == '__main__':
    # 테스트 코드
    model = WideBranchNet(frame_num=7, clip_num=5)
    model.cuda()
    
    # Frame 모드 테스트
    frame_input = torch.randn(2, 1, 7, 64, 64).cuda()
    frame_temp, frame_spat = model(frame_input, mode='frame')
    print(f"Frame-level temporal output shape: {frame_temp.shape}")  # [2, 7, 7]
    print(f"Frame-level spatial output shape: {frame_spat.shape}")   # [2, 9, 9]
    
    # Clip 모드 테스트
    clip_input = torch.randn(2, 1, 35, 64, 64).cuda()
    clip_temp, clip_spat = model(clip_input, mode='clip')
    print(f"Clip-level temporal output shape: {clip_temp.shape}")    # [2, 5, 5]
    print(f"Clip-level spatial output shape: {clip_spat.shape}")     # [2, 9, 9]