import torch


def train_one_step(Gen1, Gen2, Disc1, Disc2,
                   InputA, InputB, Latent,
                   OptGen1, OptGen2, OptDisc1, OptDisc2,
                   criterion1, criterion2):

    real_label = torch.full((InputA.size(0),), 1, dtype=torch.float)
    fake_label = torch.full((InputA.size(0),), 0, dtype=torch.float)

    ## Generate Images

    FakeImageB = Gen1(InputA, Latent)
    FakeImageA = Gen2(FakeImageB, Latent)

    ## Train A to B Disc

    Disc1.zero_grad()

    PredictionFakeImageB = Disc1(FakeImageB)
    PredictionRealImageB = Disc1(InputB)

    Disc1ErrReal = criterion1(PredictionRealImageB, real_label)
    Disc1ErrReal.backward()

    Disc1ErrFake = criterion1(PredictionFakeImageB, fake_label)
    Disc1ErrFake.backward()

    D1Loss = Disc1ErrReal + Disc1ErrFake

    OptDisc1.step()

    ## Train B to A Disc

    Disc2.zero_grad()

    PredictionFakeImageA= Disc2(FakeImageA)
    PredictionRealImageA = Disc2(InputA)

    Disc2ErrReal = criterion1(PredictionRealImageA, real_label)
    Disc2ErrReal.backward()

    Disc2ErrFake = criterion1(PredictionFakeImageA, fake_label)
    Disc2ErrFake.backward()

    D2Loss = Disc2ErrReal + Disc2ErrFake

    OptDisc2.step()

    ## Train A to B Gen

    Gen1.zero_grad()

    PredictionFakeImageB = Disc1(FakeImageB)

    Disc1ErrFake = criterion1(PredictionFakeImageB, fake_label)
    Disc1ErrFake.backward()

    ConsistencyLoss1 = criterion2(FakeImageB, InputB)
    ConsistencyLoss1.backward()

    G1Loss = Disc1ErrFake + ConsistencyLoss1

    OptGen1.step()

    ## B to A Gen

    Gen2.zero_grad()

    PredictionFakeImageB = Disc2(FakeImageB)

    Disc2ErrFake = criterion2(PredictionFakeImageA, fake_label)
    Disc2ErrFake.backward()

    ConsistencyLoss2 = criterion2(FakeImageA, InputA)
    ConsistencyLoss2.backward()

    G2Loss = Disc2ErrFake + ConsistencyLoss2

    OptGen2.step()

    return G1Loss.item(), D1Loss.item(), G2Loss.item(), D2Loss.item()






