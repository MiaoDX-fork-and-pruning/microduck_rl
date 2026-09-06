import torch

from mjlab_microduck.generalist_model import FiLMG0Actor, G0MultiHeadActor, GatedAdapterG0Actor, RoutedG0TeacherActor, build_actor


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
    actor = RoutedG0TeacherActor(*(Constant(float(i)) for i in range(6))); x = torch.zeros(6, 71)
    for i in range(6): x[i, 48 + i] = 1
    assert torch.all(actor(x)[:, 0] == torch.arange(6, dtype=torch.float32))


def test_gated_adapter_has_one_shared_action_head_and_condition_gate():
    model = GatedAdapterG0Actor(bounded=False, hidden_dim=8, adapter_dim=2)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.gate.weight[:3, :3] = torch.eye(3)
        for index, adapter in enumerate(model.adapter_up):
            adapter.bias.fill_(float(index + 1))
        model.action_head.weight[0, :] = 1.0
    x = torch.zeros((3, 71))
    x[0, 48] = 1.0
    x[1, 49] = 1.0
    x[2, 50] = 1.0
    output = model(x)
    assert output.shape == (3, 14)
    assert output[0, 0] < output[1, 0] < output[2, 0]
    assert sum(name == "action_head.weight" for name, _ in model.named_parameters()) == 1


def test_build_actor_restores_gated_adapter_metadata():
    model = build_actor({"model_kind": "gated_adapter", "bounded_actions": True,
                         "hidden_dim": 16, "adapter_dim": 4, "behavior_count": 3})
    assert isinstance(model, GatedAdapterG0Actor)
    assert model.trunk[0].in_features == 71
    assert model.action_head.out_features == 14


def test_film_actor_has_one_shared_action_head_and_restores_metadata():
    model = FiLMG0Actor(bounded=True, hidden_dim=16, output_hidden_dim=8)
    x = torch.zeros((3, 71))
    x[0, 48] = 1.0
    x[1, 49] = 1.0
    x[2, 50] = 1.0
    output = model(x)
    assert output.shape == (3, 14)
    assert torch.all(output <= 1.0) and torch.all(output >= -1.0)
    restored = build_actor({"model_kind": "film", "bounded_actions": True,
                            "hidden_dim": 16, "output_hidden_dim": 8,
                            "behavior_count": 3})
    assert isinstance(restored, FiLMG0Actor)
    assert sum(name == "action_head.weight" for name, _ in restored.named_parameters()) == 1
