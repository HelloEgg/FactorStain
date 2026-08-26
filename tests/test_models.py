import torch

from factorstain.models.factor_adapter import FactorAdapter
from factorstain.models.factorstain import FactorStain
from factorstain.models.joint_baseline import JointConditionalGenerator, ParallelFactorGenerator
from factorstain.models.mil import ABMIL
from factorstain.models.reverse_baseline import ReverseFactorStain


def test_renderers_preserve_image_shape_and_range():
    source = torch.rand(2, 3, 32, 32)
    stain, scanner = torch.tensor([0, 2]), torch.tensor([1, 0])
    for model in (FactorStain(3, 2, width=16), JointConditionalGenerator(3, 2, width=16), ParallelFactorGenerator(3, 2, width=16), ReverseFactorStain(3, 2, width=16)):
        output = model(source, stain, scanner)
        assert output.shape == source.shape
        assert 0 <= output.min() and output.max() <= 1


def test_adapter_and_mil_shapes():
    adapter = FactorAdapter(64, 4, 3, 2, bottleneck_dim=16)
    result = adapter(torch.randn(8, 64))
    assert result["features"].shape == (8, 64)
    mil = ABMIL(64, 32)
    output = mil(torch.randn(2, 10, 64))
    assert output["logits"].shape == (2, 2)
    assert torch.allclose(output["attention"].sum(1), torch.ones(2), atol=1e-5)

