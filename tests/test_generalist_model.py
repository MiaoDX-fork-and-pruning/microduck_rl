import torch

from mjlab_microduck.generalist_model import G0MultiHeadActor


def test_multihead_routes_by_behavior_condition():
    model = G0MultiHeadActor(bounded=False)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.heads[0].bias.fill_(1.0)
        model.heads[1].bias.fill_(2.0)
    x = torch.zeros((2, 71))
    x[0, 48] = 1.0
    x[1, 49] = 1.0
    output = model(x)
    assert torch.all(output[0] == 1.0)
    assert torch.all(output[1] == 2.0)
