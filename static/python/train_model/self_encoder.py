import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import os
from torch.utils.data import DataLoader, Dataset


# 定义数据集类
class ImagePairDataset(Dataset):
    def __init__(self, visible_images, infrared_images):
        self.visible_images = visible_images
        self.infrared_images = infrared_images

    def __len__(self):
        return len(self.visible_images)

    def __getitem__(self, idx):
        visible_img = cv2.imread(self.visible_images[idx], cv2.IMREAD_GRAYSCALE)
        infrared_img = cv2.imread(self.infrared_images[idx], cv2.IMREAD_GRAYSCALE)

        # 确保图像被正确读取
        if visible_img is None or infrared_img is None:
            raise ValueError(
                f"Image not found: {self.visible_images[idx]} or {self.infrared_images[idx]}"
            )

        visible_img = visible_img.astype(np.float32) / 255.0
        infrared_img = infrared_img.astype(np.float32) / 255.0

        return torch.from_numpy(visible_img).unsqueeze(0), torch.from_numpy(
            infrared_img
        ).unsqueeze(0)


# 定义DenseFuse模型
class DenseFuse(nn.Module):
    def __init__(self):
        super(DenseFuse, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def encoder(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

    def decoder(self, x):
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.conv6(x)
        return x


# 训练模型
def train_model(
    visible_images, infrared_images, num_epochs=100, batch_size=16, learning_rate=0.001
):
    dataset = ImagePairDataset(visible_images, infrared_images)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DenseFuse()
    criterion = nn.MSELoss()  # 使用均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for visible_tensor, infrared_tensor in dataloader:
            optimizer.zero_grad()

            # 特征提取
            visible_features = model.encoder(visible_tensor)
            infrared_features = model.encoder(infrared_tensor)

            # 自适应权重融合
            fusion_weight = torch.sigmoid(visible_features + infrared_features)
            fused_features = (
                fusion_weight * visible_features
                + (1 - fusion_weight) * infrared_features
            )

            # 解码融合特征
            fused_image = model.decoder(fused_features)

            # 计算损失并反向传播
            loss = criterion(fused_image, visible_tensor)  # 假设使用可见光图作为目标
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # 保存模型权重
    torch.save(model.state_dict(), "model_weight.pkl")


# 示例使用
if __name__ == "__main__":
    visible_images_path = "./static/data/visible"  # 替换为可见光图像路径
    infrared_images_path = "./static/data/infrared"  # 替换为红外图像路径

    # 收集图像文件名
    visible_images = [
        os.path.join(visible_images_path, fname)
        for fname in os.listdir(visible_images_path)
        if fname.endswith((".png", ".jpg", ".jpeg"))
    ]
    infrared_images = [
        os.path.join(infrared_images_path, fname)
        for fname in os.listdir(infrared_images_path)
        if fname.endswith((".png", ".jpg", ".jpeg"))
    ]

    # 确保可见光和红外图像数量一致
    if len(visible_images) != len(infrared_images):
        raise ValueError(
            "The number of visible images and infrared images must be the same."
        )

    train_model(
        visible_images,
        infrared_images,
        num_epochs=100,
        batch_size=16,
        learning_rate=0.001,
    )
