"""Tests for MoE layer and MoE model training stability."""

import torch
import torch.nn.functional as F
import pytest

from src.models.improved_mamba3 import MoELayer, TinyImprovedMamba3


class TestMoELayer:
    """Unit tests for the MoE layer in isolation."""

    def setup_method(self):
        torch.manual_seed(42)
        self.d_model = 64
        self.d_expert = 128
        self.n_experts = 4
        self.top_k = 2
        self.layer = MoELayer(self.d_model, self.d_expert, self.n_experts, self.top_k)

    def test_output_shape(self):
        x = torch.randn(2, 16, self.d_model)
        out = self.layer(x)
        assert out.shape == x.shape

    def test_output_finite(self):
        x = torch.randn(2, 16, self.d_model)
        out = self.layer(x)
        assert torch.isfinite(out).all(), f"Non-finite outputs: {out[~torch.isfinite(out)]}"

    def test_aux_loss_exists_and_finite(self):
        x = torch.randn(2, 16, self.d_model)
        self.layer(x)
        assert self.layer.aux_loss is not None
        assert torch.isfinite(self.layer.aux_loss), f"aux_loss={self.layer.aux_loss}"
        assert self.layer.aux_loss.item() >= 0

    def test_gradients_finite(self):
        x = torch.randn(2, 16, self.d_model, requires_grad=True)
        out = self.layer(x)
        loss = out.sum() + self.layer.aux_loss
        loss.backward()

        for name, p in self.layer.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(p.grad).all(), f"Non-finite gradient for {name}: norm={p.grad.norm()}"

    def test_gradient_magnitudes_reasonable(self):
        """Gradient norms should not be astronomically large."""
        x = torch.randn(2, 16, self.d_model)
        out = self.layer(x)
        loss = out.mean() + self.layer.aux_loss
        loss.backward()

        for name, p in self.layer.named_parameters():
            grad_norm = p.grad.norm().item()
            assert grad_norm < 1e6, f"Gradient too large for {name}: {grad_norm:.2e}"

    def test_expert_weights_sparse(self):
        """Only top_k experts should have nonzero weights per token."""
        x = torch.randn(2, 16, self.d_model)

        # Run forward to get routing
        router_logits = self.layer.router(x)
        router_probs = F.softmax(router_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_weights = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)
        one_hot = F.one_hot(top_k_indices, self.n_experts).float()
        expert_weights = (top_k_weights.unsqueeze(-1) * one_hot).sum(dim=2)

        # Each token should have exactly top_k nonzero experts
        nonzero_per_token = (expert_weights > 0).sum(dim=-1)
        assert (nonzero_per_token == self.top_k).all()

    def test_expert_weights_sum_to_one(self):
        """Expert weights per token should sum to 1."""
        x = torch.randn(2, 16, self.d_model)

        router_logits = self.layer.router(x)
        router_probs = F.softmax(router_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_weights = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)
        one_hot = F.one_hot(top_k_indices, self.n_experts).float()
        expert_weights = (top_k_weights.unsqueeze(-1) * one_hot).sum(dim=2)

        sums = expert_weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_different_inputs_different_routing(self):
        """Different inputs should generally route to different experts."""
        torch.manual_seed(123)
        x1 = torch.randn(1, 8, self.d_model)
        x2 = torch.randn(1, 8, self.d_model) * 5  # very different scale

        logits1 = self.layer.router(x1)
        logits2 = self.layer.router(x2)
        idx1 = torch.topk(logits1, self.top_k, dim=-1).indices
        idx2 = torch.topk(logits2, self.top_k, dim=-1).indices

        # Not all routing decisions should be identical
        assert not torch.equal(idx1, idx2)


class TestMoEModel:
    """Integration tests for the full model with MoE."""

    def setup_method(self):
        torch.manual_seed(42)
        self.model = TinyImprovedMamba3(
            vocab_size=256,
            d_model=64,
            n_layers=2,
            d_state=16,
            expand_factor=2,
            chunk_size=16,
            mlp_factor=4,
            n_experts=4,
            moe_top_k=2,
            moe_d_expert=128,
        )

    def test_forward_pass_finite(self):
        input_ids = torch.randint(0, 256, (2, 32))
        labels = torch.randint(0, 256, (2, 32))
        output = self.model(input_ids, labels=labels)
        assert torch.isfinite(output.loss), f"Loss is {output.loss}"
        assert torch.isfinite(output.logits).all()

    def test_backward_pass_finite(self):
        input_ids = torch.randint(0, 256, (2, 32))
        labels = torch.randint(0, 256, (2, 32))
        output = self.model(input_ids, labels=labels)
        output.loss.backward()

        total_norm = 0.0
        for name, p in self.model.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.norm().item()
                assert torch.isfinite(p.grad).all(), f"Non-finite grad for {name}"
                total_norm += param_norm ** 2
        total_norm = total_norm ** 0.5
        assert total_norm < 1e8, f"Total grad norm too large: {total_norm:.2e}"

    def test_loss_decreases_over_steps(self):
        """Model should be able to overfit a single batch."""
        input_ids = torch.randint(0, 256, (2, 32))
        labels = input_ids.clone()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            output = self.model(input_ids, labels=labels)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            losses.append(output.loss.item())

        assert losses[-1] < losses[0], f"Loss didn't decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        assert torch.isfinite(torch.tensor(losses[-1])), f"Final loss is {losses[-1]}"

    def test_muon_training_stable(self):
        """Training with Muon optimizer should not produce nan/inf."""
        from src.training.optimizer import Muon

        input_ids = torch.randint(0, 256, (2, 32))
        labels = input_ids.clone()
        optimizer = Muon(self.model.parameters(), lr=5e-4, momentum=0.95)

        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            output = self.model(input_ids, labels=labels)
            output.loss.backward()
            optimizer.step()
            losses.append(output.loss.item())

        for i, loss in enumerate(losses):
            assert not (loss != loss), f"NaN loss at step {i}"  # nan != nan
            assert abs(loss) < 1e6, f"Loss exploded at step {i}: {loss}"

    def test_moe_aux_loss_included(self):
        """The model loss should include MoE auxiliary loss."""
        input_ids = torch.randint(0, 256, (2, 32))
        labels = torch.randint(0, 256, (2, 32))

        output = self.model(input_ids, labels=labels)

        # Collect aux losses
        aux_total = 0.0
        for layer in self.model.layers:
            if isinstance(layer.mlp, MoELayer) and layer.mlp.aux_loss is not None:
                aux_total += layer.mlp.aux_loss.item()

        assert aux_total > 0, "MoE aux loss should be positive"

    def test_no_moe_model_unaffected(self):
        """A model with n_experts=0 should work identically to before."""
        model = TinyImprovedMamba3(
            vocab_size=256, d_model=64, n_layers=2, d_state=16,
            expand_factor=2, chunk_size=16, mlp_factor=4,
            n_experts=0,
        )
        input_ids = torch.randint(0, 256, (2, 32))
        labels = torch.randint(0, 256, (2, 32))
        output = model(input_ids, labels=labels)
        assert torch.isfinite(output.loss)


class TestMuonWith1DParams:
    """Test that Muon handles 1D params without explosion."""

    def test_1d_update_bounded(self):
        from src.training.optimizer import Muon

        # Create a 1D param with huge gradient
        p = torch.nn.Parameter(torch.ones(100))
        optimizer = Muon([p], lr=1e-3, momentum=0.95)

        # Simulate huge gradient
        p.grad = torch.randn(100) * 1e6
        optimizer.step()

        assert torch.isfinite(p).all(), "1D param became non-finite after huge gradient"
        # With normalized momentum, step should be bounded by lr
        max_change = (p - 1.0).abs().max().item()
        assert max_change < 0.1, f"1D param changed too much: {max_change}"

    def test_1d_update_direction_preserved(self):
        from src.training.optimizer import Muon

        p = torch.nn.Parameter(torch.zeros(10))
        optimizer = Muon([p], lr=0.01, momentum=0.0)

        # All-positive gradient should push p negative
        p.grad = torch.ones(10)
        optimizer.step()

        assert (p < 0).all(), "1D param should move opposite to gradient"


class TestMoEProductionScale:
    """Test MoE at near-production dimensions to catch scale-dependent issues."""

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)
        # Production-like: d_model=416, but fewer layers to keep test fast
        self.model = TinyImprovedMamba3(
            vocab_size=256,
            d_model=416,
            n_layers=4,
            d_state=32,
            expand_factor=2,
            chunk_size=64,
            mlp_factor=4,
            n_experts=4,
            moe_top_k=2,
            moe_d_expert=1024,
        )

    def test_forward_finite_at_scale(self):
        input_ids = torch.randint(0, 256, (8, 256))
        labels = torch.randint(0, 256, (8, 256))
        output = self.model(input_ids, labels=labels)
        assert torch.isfinite(output.loss), f"Loss={output.loss}"
        assert torch.isfinite(output.logits).all()

    def test_backward_finite_at_scale(self):
        input_ids = torch.randint(0, 256, (8, 256))
        labels = torch.randint(0, 256, (8, 256))
        output = self.model(input_ids, labels=labels)
        output.loss.backward()

        for name, p in self.model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite grad: {name}, norm={p.grad.norm()}"

    def test_muon_10_steps_no_nan(self):
        """10 Muon steps at production scale must not produce NaN."""
        from src.training.optimizer import Muon

        optimizer = Muon(self.model.parameters(), lr=5e-4, momentum=0.95)
        input_ids = torch.randint(0, 256, (4, 64))
        labels = input_ids.clone()

        for step in range(10):
            optimizer.zero_grad()
            output = self.model(input_ids, labels=labels)
            output.loss.backward()
            optimizer.step()
            loss_val = output.loss.item()
            assert loss_val == loss_val, f"NaN at step {step}"
            assert abs(loss_val) < 100, f"Loss exploded at step {step}: {loss_val}"
