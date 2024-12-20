import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm


class SampleModel(nn.Module):
    def __init__(self, intput_size: int, hidden_size: int, output_size: int):
        super(SampleModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(intput_size, hidden_size),
            nn.ReLU(),  # nn.Sigmoid()
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),  # nn.Sigmoid()
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            # nn.Linear(intput_size, output_size),
        )

    def forward(self, x):
        return self.layer(x)


class SampleDataset(torch.utils.data.Dataset):
    def __init__(self, datax, datay):
        datax = torch.tensor(datax, dtype=torch.float32)
        datay = torch.tensor(datay, dtype=torch.float32)
        self.data = list(zip(datax, datay))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(filename: str) -> tuple[list, list]:
    data = pd.read_csv(filename)
    # takes the second column as output
    y = data.iloc[:, 1].values.tolist()

    # takes the third to last column as input
    x = data.iloc[:, 2:].values.tolist()
    return x, y


if __name__ == "__main__":
    lr = 0.001
    batch_size = 512
    num_epochs = 20
    seed = 0
    momentum = 0
    set_seed(seed)

    device = "cuda"

    input_size = 10
    hidden_size = 128
    output_size = 1

    model = SampleModel(input_size, hidden_size, output_size).to(device)
    raw_data = load_data("../data/processed_data/weatherHistory.csv")
    dataset = SampleDataset(datax=raw_data[0], datay=raw_data[1])
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.MSELoss()

    pbar = tqdm(total=num_epochs * len(dataloader), dynamic_ncols=True)
    loss_list = []

    # print("Initial Weight and Bias")
    # print(model.layer[0].weight)
    # print(model.layer[0].bias)

    avg_time = 0
    for epoch in range(num_epochs):
        start_time = pd.Timestamp.now()
        total_loss = 0
        for data in dataloader:
            data = (data[0].to(device), data[1].to(device))
            optimizer.zero_grad()
            output = model(data[0]).squeeze()
            loss = criterion(output, data[1])
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss.item():.4f}")

            # print weight and bias
            # print("Weight and Bias")
            # print(model.layer[0].weight)
            # print(model.layer[0].bias)

        end_time = pd.Timestamp.now()
        epoch_time = (end_time - start_time).total_seconds()
        avg_time += epoch_time

        tqdm.write(
            f"Epoch: {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}, Time: {epoch_time:.2f} seconds"
        )
        loss_list.append(total_loss / len(dataloader))

    pbar.close()
    print(f"Average time per epoch: {avg_time / num_epochs:.2f} seconds")
    # plot loss by plt
    # plt.plot(loss_list)
    # plt.show()
