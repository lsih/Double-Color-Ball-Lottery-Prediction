# gan_predictor.py

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np


# ==============================================
# 1) 读取CSV并预处理: 红球升序 + 归一化
# ==============================================
def load_and_normalize_data(csv_file="History.csv"):
    """
    从 csv_file 读取双色球数据，并将 6 个红球升序排列后再归一化到 [0..1]。
    红球范围 [1..33]，蓝球 [1..16]。
    返回 array shape = (N, 7)：
      前6列 -> 升序红球 / 33
      第7列 -> 蓝球 / 16
    """
    df = pd.read_csv(csv_file, encoding="utf-8")

    # 如果列名是 "Blue" 大写，则重命名为 "blue"
    if "Blue" in df.columns and "blue" not in df.columns:
        df = df.rename(columns={"Blue": "blue"})
    # 如果列名是 "Red1..Red6" 大写，则重命名为小写
    for i in range(1, 7):
        if f"Red{i}" in df.columns and f"red{i}" not in df.columns:
            df = df.rename(columns={f"Red{i}": f"red{i}"})

    # 确保有 red1..red6, blue 这些列
    required_cols = [f"red{i}" for i in range(1, 7)] + ["blue"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"CSV中缺少必要列: {c}")

    # 收集所有期，并对红球进行升序处理
    all_draws = []
    for _, row in df.iterrows():
        reds = [row[f"red{i}"] for i in range(1, 7)]
        reds.sort()  # 升序
        blue = row["blue"]
        draw = reds + [blue]
        all_draws.append(draw)

    all_draws = np.array(all_draws, dtype=np.float32)  # shape (N,7)

    # 红球除以 33，蓝球除以 16
    all_draws[:, :6] /= 33.0
    all_draws[:, 6] /= 16.0
    return all_draws


# ==============================================
# 2) 定义生成器 + 判别器
# ==============================================
class Generator(nn.Module):
    def __init__(self, z_dim, data_dim=7):
        """
        这里 data_dim=7: 前6列(红球), 第7列(蓝球)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, data_dim),
            nn.Sigmoid(),  # 输出范围 [0..1]
        )

    def forward(self, z):
        # 先得到 (batch,7)
        out = self.net(z)  # shape: (batch_size,7)
        # 然后对前6列(红球)做升序 sort，避免出现无序
        red_part = out[:, :6]
        sorted_red, _ = torch.sort(red_part, dim=1)  # 按行排序
        # 剩下第7列是蓝球
        blue_part = out[:, 6:].clone()
        # 拼回 (batch,7)
        out = torch.cat([sorted_red, blue_part], dim=1)
        return out


class Discriminator(nn.Module):
    def __init__(self, data_dim=7):
        """
        判别器假设输入中，前6维为升序红球，最后1维为蓝球
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim, 64), nn.LeakyReLU(0.2), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# ==============================================
# 3) 训练并保存
# ==============================================
def train_gan_and_save(
    csv_file="History.csv",
    model_dir="gan_models",
    generator_file="G.pth",
    discriminator_file="D.pth",
    num_epochs=1000,
    batch_size=32,
    z_dim=16,
):
    """
    训练GAN, 并将 生成器G + 判别器D 的权重保存到本地.
    """
    os.makedirs(model_dir, exist_ok=True)

    # 读取+归一化
    all_draws_norm = load_and_normalize_data(csv_file)
    data_tensor = torch.tensor(all_draws_norm, dtype=torch.float32)

    # 构建
    G = Generator(z_dim=z_dim, data_dim=7)
    D = Discriminator(data_dim=7)

    criterion = nn.BCELoss()
    g_optimizer = torch.optim.Adam(G.parameters(), lr=1e-4)
    d_optimizer = torch.optim.Adam(D.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        # 从 data_tensor 抽取 batch_size 条
        idx = np.random.randint(0, len(data_tensor), batch_size)
        real_data = data_tensor[idx]  # (batch_size,7)
        real_labels = torch.ones(batch_size, 1)

        # 生成假数据
        z = torch.randn(batch_size, z_dim)
        fake_data = G(z)
        fake_labels = torch.zeros(batch_size, 1)

        # ---- 训练判别器D ----
        d_optimizer.zero_grad()
        d_real = D(real_data)
        d_fake = D(fake_data.detach())
        d_loss_real = criterion(d_real, real_labels)
        d_loss_fake = criterion(d_fake, fake_labels)
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()

        # ---- 训练生成器G ----
        g_optimizer.zero_grad()
        d_fake_gen = D(fake_data)
        g_loss = criterion(d_fake_gen, real_labels)
        g_loss.backward()
        g_optimizer.step()

        # 打印
        if (epoch + 1) % 200 == 0 or epoch == num_epochs - 1:
            print(
                f"Epoch [{epoch+1}/{num_epochs}] d_loss={d_loss.item():.4f}, g_loss={g_loss.item():.4f}"
            )

    # 保存
    torch.save(G.state_dict(), os.path.join(model_dir, generator_file))
    torch.save(D.state_dict(), os.path.join(model_dir, discriminator_file))
    print(f"GAN训练完成: G -> {generator_file}, D -> {discriminator_file}")


# ==============================================
# 4) 加载G后, 生成号码. 并做"若重复则重采"的处理.
# ==============================================
def load_gan_and_generate_norepeat(
    model_dir="gan_models",
    generator_file="G.pth",
    num_samples=1,
    z_dim=16,
    max_attempts=50,
):
    """
    从本地加载生成器G, 并生成 num_samples 组 (6红+1蓝),
    如果红球出现重复就放弃这一组, 重采. 最多重采 max_attempts 次.
    返回 shape=(num_samples,7) 整型数组
    """
    # 构建同样结构
    G = Generator(z_dim=z_dim, data_dim=7)
    # 加载权重
    gen_path = os.path.join(model_dir, generator_file)
    G.load_state_dict(torch.load(gen_path))
    G.eval()

    results = []
    for _ in range(num_samples):
        for attempt in range(max_attempts):
            z = torch.randn(1, z_dim)
            with torch.no_grad():
                out = G(z).cpu().numpy()[0]  # shape (7,)

            # 反归一化: 前6列 *33, 第7列 *16
            reds = np.round(out[:6] * 33).astype(int)
            blue = int(round(out[6] * 16))
            reds = np.clip(reds, 1, 33)
            blue = np.clip(blue, 1, 16)

            # 检查是否重复
            if len(set(reds)) == 6:
                # 没有重复 => 接受
                row = np.concatenate([reds, [blue]])
                results.append(row)
                break
        else:
            # 说明 for attempt in range(max_attempts)都没成功
            # 只能"补丁"修一下
            row = fix_duplicates(np.concatenate([reds, [blue]]))
            results.append(row)

    return np.array(results, dtype=int)


def fix_duplicates(row7):
    """
    如果 row7[:6] 有重复, 强行替换掉重复红球 => 不重复
    row7: shape(7,)
    """
    reds = list(row7[:6])
    unique_reds = []
    for r in reds:
        if r not in unique_reds:
            unique_reds.append(r)
        else:
            # 找个未使用过的
            for cand in range(1, 34):
                if cand not in unique_reds:
                    unique_reds.append(cand)
                    break
    unique_reds = unique_reds[:6]
    unique_reds.sort()
    newrow = np.array(unique_reds + [row7[6]], dtype=int)
    return newrow


# ==============================================
# 5) DEMO
# ==============================================
if __name__ == "__main__":
    # 第一步: 训练并保存
    train_gan_and_save(
        csv_file="History.csv",
        model_dir="gan_models",
        generator_file="G.pth",
        discriminator_file="D.pth",
        num_epochs=1000,
        batch_size=32,
        z_dim=16,
    )

    # 第二步: 加载后生成 3 组
    samples = load_gan_and_generate_norepeat(
        model_dir="gan_models",
        generator_file="G.pth",
        num_samples=3,
        z_dim=16,
        max_attempts=50,
    )
    for i, row in enumerate(samples):
        reds = row[:6]
        blue = row[6]
        print(f"Sample {i+1}: 红球={reds}, 蓝球={blue}")
