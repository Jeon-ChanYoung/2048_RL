import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Game2048Env:
    def __init__(self, size=4):
        self.size = size
        self.state = np.zeros((size, size), dtype=np.int32)
        self.reset()

    def reset(self):
        self.state = np.zeros((self.size, self.size), dtype=np.int32)
        self._add_tile()
        self._add_tile()
        return self.state.copy()
    
    # action: 0=left, 1=down, 2=right, 3=up
    def step(self, action):
        prev_state = self.state.copy()
        self.state, reward = self._move(action)
        done = self._is_done()
        if not np.array_equal(prev_state, self.state):
            self._add_tile()

        return self.state.copy(), reward, done

    def action_sample(self):
        return random.randint(0, 3)

    def possible_actions(self):
        res = []
        for action in range(4):
            if self.can_perform(action):
                res.append(action)
        return res

    def can_perform(self, action):
        """액션 수행 가능 여부 판단: 액션을 실제로 실행해보고 변화가 있는지 확인"""
        tmp_state = self.state.copy()
        tmp_state, _ = self._simulate_move(tmp_state, action)
        return not np.array_equal(self.state, tmp_state)  # 원래 상태와 비교하여 변화가 있으면 가능

    def _add_tile(self):
        empty = list(zip(*np.where(self.state == 0)))
        if empty:
            r, c = random.choice(empty)
            self.state[r, c] = 2 if random.random() < 0.9 else 4

    def _simulate_move(self, tmp_state, action):
        rotated = np.rot90(tmp_state, action)  # 회전해서 좌측 이동으로 통일
        new_board = np.zeros_like(rotated)

        for i in range(self.size):
            row = rotated[i][rotated[i] != 0]  # 0 제거
            new_row = []
            skip = False
            for j in range(len(row)):
                if skip:
                    skip = False
                    continue
                if j + 1 < len(row) and row[j] == row[j + 1]:
                    new_val = row[j] * 2
                    new_row.append(new_val)
                    skip = True
                else:
                    new_row.append(row[j])
            new_board[i, :len(new_row)] = new_row

        # 다시 회전 복구
        new_state = np.rot90(new_board, -action)

        return new_state, 0
    
    def _move(self, action):
        rotated = np.rot90(self.state, action)  # 회전해서 좌측 이동으로 통일
        new_board = np.zeros_like(rotated)
        max_tile_before = self.state.max()

        for i in range(self.size):
            row = rotated[i][rotated[i] != 0]  # 0 제거
            new_row = []
            skip = False
            for j in range(len(row)):
                if skip:
                    skip = False
                    continue
                if j + 1 < len(row) and row[j] == row[j + 1]:
                    new_val = row[j] * 2
                    new_row.append(new_val)
                    skip = True
                else:
                    new_row.append(row[j])
            new_board[i, :len(new_row)] = new_row

        # 다시 회전 복구
        new_state = np.rot90(new_board, -action)

        # 보상 계산
        # reward = evaluation(new_state, n_empty)
        reward = 0

        return new_state, reward

    def _is_done(self):
        if np.any(self.state == 0):
            return False
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.state[i, j] == self.state[i, j + 1]:
                    return False
                if self.state[j, i] == self.state[j + 1, i]:
                    return False
        return True
    
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        # First layer convolution layers
        self.conv1 = nn.Conv2d(16, 128, kernel_size=(1, 2))
        self.conv2 = nn.Conv2d(16, 128, kernel_size=(2, 1))

        # Second layer convolution layers
        self.conv11 = nn.Conv2d(128, 128, kernel_size=(1, 2))
        self.conv12 = nn.Conv2d(128, 128, kernel_size=(2, 1))
        self.conv21 = nn.Conv2d(128, 128, kernel_size=(1, 2))
        self.conv22 = nn.Conv2d(128, 128, kernel_size=(2, 1))

        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(self._get_flattened_size(), 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        # Pass through convolutional layers
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x11 = F.relu(self.conv11(x1))
        x12 = F.relu(self.conv12(x1))
        x21 = F.relu(self.conv21(x2))
        x22 = F.relu(self.conv22(x2))

        # Flatten the outputs of the convolutional layers
        x1 = x1.view(x1.size(0), -1)  # Flatten for x1
        x2 = x2.view(x2.size(0), -1)  # Flatten for x2
        x11 = x11.view(x11.size(0), -1)  # Flatten for x11
        x12 = x12.view(x12.size(0), -1)  # Flatten for x12
        x21 = x21.view(x21.size(0), -1)  # Flatten for x21
        x22 = x22.view(x22.size(0), -1)  # Flatten for x22

        # Concatenate all flattened outputs
        concat = torch.cat((x1, x2, x11, x12, x21, x22), dim=1)
        return self.fc(concat)
    
    def _get_flattened_size(self):
        # Assuming input size of (batch_size, 16, 4, 4)
        dummy_input = torch.zeros(1, 16, 4, 4)
        x1 = F.relu(self.conv1(dummy_input))
        x2 = F.relu(self.conv2(dummy_input))
        x11 = F.relu(self.conv11(x1))
        x12 = F.relu(self.conv12(x1))
        x21 = F.relu(self.conv21(x2))
        x22 = F.relu(self.conv22(x2))

        # Return the total flattened size
        return (
            x1.numel() + x2.numel() +
            x11.numel() + x12.numel() +
            x21.numel() + x22.numel()
        )

def one_hot_encode(state, maxtile = 16):
    state_log = state.copy()
    non_zero_mask = state_log != 0
    state_log[non_zero_mask] = np.log2(state_log[non_zero_mask])
    one_hot_state = np.zeros((1, maxtile, 4, 4))

    for i in range(4):
        for j in range(4):
            v = state_log[i][j]
            if 0 <= v < maxtile:
                one_hot_state[0, v, i, j] = 1.0
    return one_hot_state

model_path = "main_net2048.pth"

class RLAgent:
    def __init__(self, model_path = model_path, device = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.main_net = DQN().to(self.device)
        self.main_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.main_net.eval()

    def act(self, env: Game2048Env):
        state = one_hot_encode(env.state) 
        state_tensor = torch.FloatTensor(state).to(self.device)

        with torch.no_grad():
            q_values = self.main_net(state_tensor).squeeze(0)

        possible = env.possible_actions()
        if not possible:
            return None
        q_values_np = q_values.cpu().numpy()
        mask = np.array([1 if a in possible else 0 for a in range(4)])
        q_values_np = q_values_np * mask - (1 - mask) * 1e9
        return int(q_values_np.argmax())