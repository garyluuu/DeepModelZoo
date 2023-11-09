from typing import Callable
import torch
import torch.nn as nn


def mvWin(
    inputLen,
    padval=0.0,
):
    def inFn(history):
        histlen = len(history)
        addLen = max(0, inputLen - histlen)
        fetchLen = inputLen - addLen
        padmatrix = padval * torch.ones_like(history[0])
        return torch.stack([*(padmatrix,) * addLen, *history[-fetchLen:]], dim=0)

    return inFn


def allin():
    def inFn(history):
        return torch.stack(history, dim=0)

    return inFn


def last():
    def inFn(history):
        return history[None,-1:]

    return inFn


class SeqWrapper(nn.Module):
    def __init__(
        self,
        net: nn.Module,
        # evo_net: nn.Module,
        evolveLen: int,
        hisPackFn: Callable[[list], torch.Tensor],
        hasAuxiliary: bool = False,
        outAuxiliary: bool = False,
    ) -> None:
        super().__init__()
        self.net = net
        # self.evo_net = evo_net
        self.evolveLen = evolveLen
        self.hisPackFn = hisPackFn
        self.outAuxiliary = outAuxiliary
        self.hasAuxiliary = hasAuxiliary


    def forward(
        self,
        ic,
        constant,
        AuxIn=None,
    ):
        # results: save every snapshot, of shape [B, D, H, W, C]
        results = [*ic]
        icLen = len(results)
        u = ic
        
        # h = torch.zeros(1, results[0].shape[0], 9).to(ic.device)
        # c = torch.zeros(1, results[0].shape[0], 9).to(ic.device)

        for _ in range(self.evolveLen):
            inp = self.hisPackFn(results)
            u = self.net(inp, constant, AuxIn=AuxIn)
            # u, (h,c) = self.net(inp, constant, h, c, AuxIn=AuxIn)
            # u = self.evo_net(h, constant, AuxIn=AuxIn)

            if self.hasAuxiliary:
                u, AuxIn = u, AuxIn

            results.extend(u)

        if self.outAuxiliary:
            return torch.stack(results)[icLen:], AuxIn
        else:
            return torch.stack(results)[icLen:]
