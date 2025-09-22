import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 모델 정의
class StarToGPS(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    # 저장된 데이터셋 불러오기
    samples = torch.load("star_gps_dataset1.pt")
    dataset = [(x, y) for x, y in samples]
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 모델 초기화
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 모델의 입력 차원을 10으로
    model = StarToGPS(input_dim=10).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 학습
    num_epochs = 50
    aa = '1'
    while aa == '1':
    # for epoch in range(num_epochs):
        running_loss = 0
        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            #Azimuth 데이터(인덱스 6)를 제거하고 모델에 입력
            x_batch = torch.cat([x_batch[:, :6], x_batch[:, 7:]], dim=1)
            
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x_batch.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Loss: {epoch_loss:.6f}")
        if epoch_loss < 1.0:

            aa = input("hi:")

    # 모델 저장
    torch.save(model.state_dict(), "star_gps_model2.pth")
    print("모델 학습 완료 및 저장됨!")