import torch
import torch.nn.functional as F

class Memory:
    def __init__(self, size=512):
        self.size = size
        self.memory = []
        self.tmp_size = 0

    def in_memory(self, data: torch.Tensor):
        data = data.detach()
        for feature in data:
            if self.tmp_size == self.size:
                self.memory.pop(0)
                self.tmp_size -= 1
            self.memory.append(feature)
            self.tmp_size += 1


class PosMemory:
    def __init__(self, size=210, length=8):
        self.size = size
        self.memory = []
        for i in range(210):
            self.memory.append([])
        self.tmp_size = [0] * size
        self.max_length = length

    def in_memory(self, data: torch.Tensor, count: int, batchsize: int):
        loss = 0
        for i in range(batchsize):
            if self.tmp_size[count + i] == self.max_length:
                self.memory[count + i].pop(0)
                self.tmp_size[count + i] -= 1
            self.memory[count + i].append(data[i])
            pos_feature = torch.cat(self.memory[count + i], dim=0).reshape(-1, 1024)
            tmp_loss = F.normalize(data[i].unsqueeze(dim=0).detach(), p=2, dim=1) @ F.normalize(pos_feature.detach(),
                                                                                                p=2, dim=1).t()
            loss += tmp_loss.mean()
            self.tmp_size[count + i] += 1
        return loss / batchsize


if __name__ == '__main__':
    pass
