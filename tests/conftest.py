import pytest
import torch

from gfg_instruments.adapters.nanogpt import GPT, GPTConfig, NanoGPTAdapter
from gfg_instruments.contracts import Batch, ExecutionContract
from gfg_instruments.evidence import Journal


@pytest.fixture(autouse=True)
def deterministic_runtime():
    torch.set_num_threads(2)
    torch.manual_seed(1234)
    torch.use_deterministic_algorithms(True)
    yield


@pytest.fixture
def batch():
    x = torch.randint(0, 7, (8, 3))
    y = torch.full_like(x, -1)
    y[:, -1] = torch.arange(8) % 7
    return Batch(tuple(f"small-mod7:{i}" for i in range(8)), (x,), y,
                 tuple(f"target:{i % 7:02d}" for i in range(8)), "generated-unit-input-v1")


@pytest.fixture
def adapter(batch):
    model = GPT(GPTConfig(block_size=3, vocab_size=7, n_layer=2, n_head=2, n_embd=8, dropout=0., bias=False))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, betas=(.9, .98), weight_decay=.1, foreach=False)
    result = NanoGPTAdapter(model, optimizer)
    result.native_update(batch, ExecutionContract())
    return result


@pytest.fixture
def journal(tmp_path):
    return Journal(tmp_path / "evidence", run_id="test-run")
