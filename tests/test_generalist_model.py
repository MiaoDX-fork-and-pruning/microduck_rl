import torch

from mjlab_microduck.generalist_model import G0MultiHeadActor, RoutedG0TeacherActor


def test_multihead_routes_by_behavior_condition():
    model = G0MultiHeadActor(bounded=False)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.heads[0].bias.fill_(1.0)
        model.heads[1].bias.fill_(2.0)
        model.heads[2].bias.fill_(3.0)
    x = torch.zeros((3, 71))
    x[0, 48] = 1.0
    x[1, 49] = 1.0
    x[2, 50] = 1.0
    output = model(x)
    assert torch.all(output[0] == 1.0)
    assert torch.all(output[1] == 2.0)
    assert torch.all(output[2] == 3.0)

def test_routed_actor_uses_behavior_bit():
    class Constant(torch.nn.Module):
        def __init__(self, value): super().__init__(); self.value = value
        def forward(self, x): return torch.full((x.shape[0], 14), self.value)
    actor = RoutedG0TeacherActor(Constant(1.0), Constant(2.0)); x = torch.zeros(2, 71); x[1, 49] = 1
    assert torch.all(actor(x)[0] == 1) and torch.all(actor(x)[1] == 2)
