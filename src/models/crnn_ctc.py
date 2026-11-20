import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Một block CNN cơ bản:
    Conv2d -> BatchNorm -> ReLU -> MaxPool
    """

    def __init__(self, in_channels, out_channels, pool_kernel=(2, 2), pool_stride=(2, 2)):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, 
                      out_channels, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=pool_kernel,
                stride=pool_stride
            )
        )

    def forward(self, x):
        return self.block(x)


class CRNNCTC(nn.Module):
    """
    CRNN + CTC cho nhận diện chữ viết tay tiếng Việt mức dòng.

    Input:
        images: [B, 1, 64, W]

    Output:
        logits: [T, B, num_classes]

    Ý tưởng:
        CNN trích đặc trưng ảnh
        -> biến feature map thành chuỗi theo chiều ngang
        -> BiLSTM học quan hệ trái-phải
        -> Linear phân loại ký tự tại từng timestep
    """

    def __init__(
        self,
        num_classes,
        input_channels=1,
        hidden_size=256,
        num_lstm_layers=2,
        dropout=0.3
    ):
        super().__init__()

        self.num_classes = num_classes
        self.hidden_size = hidden_size

        # Input: [B, 1, 64, W]
        self.cnn = nn.Sequential(
            # [B, 1, 64, W] -> [B, 64, 32, W/2]
            ConvBlock(input_channels, 64, pool_kernel=(2, 2), pool_stride=(2, 2)),

            # [B, 64, 32, W/2] -> [B, 128, 16, W/4]
            ConvBlock(64, 128, pool_kernel=(2, 2), pool_stride=(2, 2)),

            # [B, 128, 16, W/4] -> [B, 256, 8, W/4]
            # Chỉ giảm height, không giảm width
            ConvBlock(128, 256, pool_kernel=(2, 1), pool_stride=(2, 1)),

            # [B, 256, 8, W/4] -> [B, 512, 4, W/4]
            ConvBlock(256, 512, pool_kernel=(2, 1), pool_stride=(2, 1)),

            # [B, 512, 4, W/4] -> [B, 512, 1, W/4]
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, None)),
        )

        # Sau CNN:
        # [B, 512, 1, T] -> [B, T, 512]
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )

        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, images):
        """
        images: [B, 1, 64, W]
        return logits: [T, B, num_classes]
        """

        features = self.cnn(images)
        # features: [B, 512, 1, T]

        features = features.squeeze(2)
        # [B, 512, T]

        features = features.permute(0, 2, 1)
        # [B, T, 512]

        rnn_out, _ = self.rnn(features)
        # [B, T, hidden_size * 2]

        logits = self.classifier(rnn_out)
        # [B, T, num_classes]

        logits = logits.permute(1, 0, 2)
        # [T, B, num_classes]

        return logits

    def get_output_lengths(self, image_widths):
        """
        Tính độ dài sequence sau CNN.

        Vì CNN giảm width 2 lần ở block 1 và 2:
            W -> W/2 -> W/4

        Các block sau chỉ giảm height, không giảm width.

        image_widths: Tensor [B]
        return: Tensor [B]
        """
        return image_widths // 4