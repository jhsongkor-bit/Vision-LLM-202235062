import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


train_loss_history = [0.3167, 0.0943, 0.0665]
test_loss_history = [0.1030, 0.0654, 0.0509]

train_accuracy_history = [0.9042, 0.9713, 0.9798]
test_accuracy_history = [0.9669, 0.9794, 0.9831]


def show_feature_maps(feature_maps, cmap="viridis", num_maps=None):
    feature_maps = feature_maps.squeeze(0).detach().cpu()

    if num_maps is None:
        num_maps = feature_maps.shape[0]

    plt.figure(figsize=(12, 4))

    for i in range(min(num_maps, feature_maps.shape[0])):
        plt.subplot(1, num_maps, i + 1)
        plt.imshow(feature_maps[i], cmap=cmap)
        plt.axis("off")

    plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 불러오기
loaded_model = SimpleCNN()

state_dict = torch.load(
    "../src/models/MNIST/MNIST_CNN.pth",
    map_location="cpu",
    weights_only=True
)

loaded_model.load_state_dict(state_dict)
loaded_model = loaded_model.to(device)
loaded_model.eval()

test_dataset = datasets.MNIST(
    root="../data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)


# TODO: plt.plot(x축_데이터, y축_데이터, marker="o", label="데이터_이름")을 사용하여 그래프 생성
plt.figure(figsize=(8, 5))

# TODO: train_loss_history 그래프 생성
plt.plot(range(1, 4), train_loss_history, marker="o", label="Train Loss")

# TODO: test_loss_history 그래프 생성
plt.plot(range(1, 4), test_loss_history, marker="o", label="Test Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN Loss")
plt.grid()
plt.legend()
plt.show()

# TODO: plt.plot(x축_데이터, y축_데이터, marker="o", label="데이터_이름")을 사용하여 그래프 생성
plt.figure(figsize=(8, 5))

# TODO: train_accuracy_history 그래프 생성 (백분율)
plt.plot(range(1, 4), [x * 100 for x in train_accuracy_history],
         marker="o", label="Train Accuracy")

# TODO: test_accuracy_history 그래프 생성 (백분율)
plt.plot(range(1, 4), [x * 100 for x in test_accuracy_history],
         marker="o", label="Test Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("CNN Accuracy")
plt.grid()
plt.legend()
plt.show()

# TODO: CNN 각 층의 Feature Map 시각화
# TODO: cmap="viridis" 사용

# TODO: 이미지 선택
image, true_label = test_dataset[0]
input_batch = image.unsqueeze(0).to(device)

# TODO: 역전파(파라미터 수정) 없이 순전파 수행
loaded_model.eval()

with torch.no_grad():
    conv1 = loaded_model.conv1(input_batch)
    relu1 = loaded_model.relu(conv1)
    pool1 = loaded_model.pool(relu1)

    conv2 = loaded_model.conv2(pool1)
    relu2 = loaded_model.relu(conv2)
    pool2 = loaded_model.pool(relu2)

# TODO: Conv1 Feature Maps 8개 시각화
show_feature_maps(conv1, cmap="viridis")

# TODO: Conv2 Feature Maps 16개 시각화
show_feature_maps(conv2, cmap="viridis")

# TODO: 6개 Layer별 대표 Feature Map 4개씩 비교
layer_names = [
    "conv1",
    "relu1",
    "pool1",
    "conv2",
    "relu2",
    "pool2"
]

feature_maps = [
    conv1,
    relu1,
    pool1,
    conv2,
    relu2,
    pool2
]

for name, feature_map in zip(layer_names, feature_maps):
    print(name)
    show_feature_maps(feature_map, cmap="viridis", num_maps=4)