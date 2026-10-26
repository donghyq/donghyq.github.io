"""AI Infra exercise: compact causal LM with single-process / DDP / BF16 / profiling.

Examples:
  python examples/train_tiny_gpt.py --steps 12
  torchrun --standalone --nproc_per_node=2 examples/train_tiny_gpt.py --steps 12
  python examples/train_tiny_gpt.py --steps 12 --profile

Dataset is synthetic; throughput and correctness of the training step are the goal.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.float()
        z = z * torch.rsqrt(z.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return z.to(x.dtype) * self.weight.to(x.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden: int, heads: int):
        super().__init__()
        assert hidden % heads == 0
        self.heads = heads
        self.head_dim = hidden // heads
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.out = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, hidden = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def shape(z):
            return z.reshape(b, s, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = shape(q), shape(k), shape(v)
        # SDPA dispatches to an available attention backend (Flash / memory efficient / math).
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        y = y.transpose(1, 2).contiguous().reshape(b, s, hidden)
        return self.out(y)


class DecoderBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, ff_mult: int = 4):
        super().__init__()
        self.attn_norm = RMSNorm(hidden)
        self.attn = CausalSelfAttention(hidden, heads)
        self.ff_norm = RMSNorm(hidden)
        ff = hidden * ff_mult
        self.gate_up = nn.Linear(hidden, ff * 2, bias=False)
        self.down = nn.Linear(ff, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        gate, up = self.gate_up(self.ff_norm(x)).chunk(2, dim=-1)
        return x + self.down(F.silu(gate) * up)


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, hidden: int, layers: int, heads: int,
                 use_checkpoint: bool = False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.blocks = nn.ModuleList(DecoderBlock(hidden, heads) for _ in range(layers))
        self.norm = RMSNorm(hidden)
        self.head = nn.Linear(hidden, vocab_size, bias=False)
        self.head.weight = self.embedding.weight  # weight tying
        self.use_checkpoint = use_checkpoint

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False) if self.use_checkpoint else block(x)
        return self.head(self.norm(x))


def init_distributed() -> tuple[torch.device, int, int, bool]:
    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cpu')
    distributed = world > 1
    if distributed:
        dist.init_process_group(backend='nccl' if device.type == 'cuda' else 'gloo')
    return device, rank, world, distributed


def lr_lambda(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return (step + 1) / max(warmup_steps, 1)
    t = (step - warmup_steps) / max(total_steps - warmup_steps - 1, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(max(t, 0), 1)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=12)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--seq-len', type=int, default=64)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--vocab', type=int, default=1024)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--grad-accum', type=int, default=1)
    p.add_argument('--bf16', action='store_true')
    p.add_argument('--checkpoint', action='store_true')
    p.add_argument('--profile', action='store_true')
    p.add_argument('--profile-dir', default='./profiler_traces')
    args = p.parse_args()
    if args.steps < 1 or args.grad_accum < 1:
        p.error('steps and grad-accum must be >= 1')
    if args.profile and args.steps < 7:
        p.error('--profile needs at least 7 steps (wait=2,warmup=2,active=3)')

    device, rank, world, distributed = init_distributed()
    torch.manual_seed(2026)
    model = TinyGPT(args.vocab, args.hidden, args.layers, args.heads,
                    use_checkpoint=args.checkpoint).to(device)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[device.index] if device.type == 'cuda' else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=(device.type == 'cuda'))
    warmup = min(4, max(1, args.steps // 4))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_lambda(s, args.steps, warmup))

    activities = [ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(ProfilerActivity.CUDA)
    prof_ctx = (profile(activities=activities, schedule=schedule(wait=2, warmup=2, active=3),
                        on_trace_ready=tensorboard_trace_handler(args.profile_dir),
                        record_shapes=False, profile_memory=True) if args.profile and rank == 0
                else nullcontext())
    if rank == 0:
        nparam = sum(x.numel() for x in model.parameters())
        print(f'device={device}, ranks={world}, params={nparam:,}, global_tokens/step='
              f'{world * args.batch_size * args.seq_len * args.grad_accum}')
    model.train()
    with prof_ctx as prof:
        for step in range(args.steps):
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            for micro in range(args.grad_accum):
                # different random samples per rank / step; not meant for convergence.
                torch.manual_seed(2026 + rank * 100000 + step * args.grad_accum + micro)
                tokens = torch.randint(args.vocab, (args.batch_size, args.seq_len + 1), device=device)
                x, target = tokens[:, :-1], tokens[:, 1:]
                sync_ctx = (model.no_sync() if distributed and micro < args.grad_accum - 1
                            else nullcontext())
                with sync_ctx:
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                        enabled=args.bf16 and device.type == 'cuda'):
                        logits = model(x)
                        loss = F.cross_entropy(logits.float().reshape(-1, args.vocab), target.reshape(-1))
                    (loss / args.grad_accum).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            if distributed:
                # Compare slowest rank; collective itself is after timing, outside step.
                max_elapsed = torch.tensor([elapsed], device=device)
                dist.all_reduce(max_elapsed, op=dist.ReduceOp.MAX)
                elapsed = max_elapsed.item()
            if rank == 0:
                global_tokens = world * args.batch_size * args.seq_len * args.grad_accum
                print(f'step={step:03d} loss={loss.item():.4f} '
                      f'step_ms={elapsed*1e3:.1f} tokens_s={global_tokens/elapsed:.1f} '
                      f'lr={scheduler.get_last_lr()[0]:.6f}')
            if prof is not None:
                prof.step()
    if device.type == 'cuda' and rank == 0:
        print(f'peak_allocated_MiB={torch.cuda.max_memory_allocated(device)/2**20:.1f}')
    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
