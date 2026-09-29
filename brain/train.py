"""Train the unified Raksam Brain V3 backbone with auditable checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, random_split

from brain.dataset import UnifiedDataset
from brain.model import BrainConfig, BrainModel
from brain.tokenizer.tokenizer import CharTokenizer

ACTIVE = Path("brain/checkpoints/latest.pt")
CANDIDATE = Path("brain/checkpoints/candidate.pt")
OPTIMIZER = Path("brain/checkpoints/optimizer.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42


def collect_public_web_data(config_path: str = "config/web_learning.yaml") -> dict:
    """Run one bounded public-web collection pass without starting another train.

    ``web_learner.continual.run_learning`` includes candidate training, so it
    cannot be used here without recursively launching this script.  The
    crawler writes provenance-preserving JSONL records to ``data/web_knowledge``;
    ``UnifiedDataset`` then picks them up in the dataset built below.
    """
    from web_learner.crawler import PublicWebLearner

    learner = PublicWebLearner(config_path)
    before = len(learner.seen)
    crawl = learner.crawl()
    return {**crawl, "new_documents": len(learner.seen) - before}


def weighted_loss(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Loss only on answer/action targets; each source example keeps its weight."""
    raw = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1, reduction="none").view_as(targets)
    mask = (targets >= 0).float()
    example_weights = weights[:, None].expand_as(raw)
    return (raw * mask * example_weights).sum() / (mask * example_weights).sum().clamp_min(1.0)


def checkpoint_payload(model, config, dataset_size, metadata):
    return {"model_state": model.state_dict(), "config": config.__dict__, "model_name": "Raksam Brain V3 (RB3)", "created_by": "Prosanto Raksam", "training": {"examples": dataset_size, "parameter_count": model.num_parameters(), **metadata}}


def validation_signature(dataset: UnifiedDataset, indices) -> str:
    """Fingerprint the exact held-out examples used for checkpoint scoring."""
    digest = hashlib.sha256()
    examples = []
    for index in indices:
        ids, labels, weight = dataset.examples[int(index)]
        examples.append((ids, labels, float(weight)))
    for item in sorted(examples, key=lambda value: json.dumps(value, separators=(",", ":"))):
        digest.update(json.dumps(item, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def evaluate(model, loader, max_steps=0):
    model.eval(); losses = []
    with torch.no_grad():
        for step, (x, y, weights) in enumerate(loader, 1):
            logits, _ = model(x.to(DEVICE), None)
            losses.append(weighted_loss(logits, y.to(DEVICE), weights.to(DEVICE)).item())
            if max_steps and step >= max_steps: break
    return sum(losses) / max(1, len(losses)), len(losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--candidate", action="store_true")
    parser.add_argument("--epochs", type=int, default=1); parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4); parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0, help="Cap training batches per epoch; makes the run partial.")
    parser.add_argument("--max-val-steps", type=int, default=0, help="Cap validation batches; reports sampled validation.")
    parser.add_argument("--save-every", type=int, default=250); parser.add_argument("--run-label", default="")
    parser.add_argument("--skip-web-collection", action="store_true", help="Do not collect configured public web data before this run.")
    args = parser.parse_args(); random.seed(SEED); torch.manual_seed(SEED)
    tokenizer = CharTokenizer()
    with open("config/settings.yaml", encoding="utf8") as file: architecture = (yaml.safe_load(file) or {}).get("brain", {}).get("architecture", {})
    with open("config/web_learning.yaml", encoding="utf8") as file: web_config = yaml.safe_load(file) or {}
    if web_config.get("enabled", False) and web_config.get("collect_before_train", False) and not args.skip_web_collection:
        try:
            collected = collect_public_web_data()
            print(f"web_collection fetched={collected['fetched']} accepted={collected['accepted']} new={collected['new_documents']}")
        except Exception as exc:
            # Training can still safely use the existing, already-auditable
            # knowledge corpus when a public source is temporarily unavailable.
            print(f"web_collection skipped: {exc}")
    dataset = UnifiedDataset(tokenizer, block_size=int(architecture.get("block_size", 256)))
    if len(dataset) < 20: raise SystemExit(f"Need at least 20 usable examples; found {len(dataset)}.")
    validation_size = max(2, int(len(dataset) * args.val_ratio))
    train_set, validation_set = random_split(dataset, [len(dataset) - validation_size, validation_size], generator=torch.Generator().manual_seed(SEED))
    heldout_signature = validation_signature(dataset, validation_set.indices)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True); validation_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
    config = BrainConfig(vocab_size=tokenizer.vocab_size, **architecture); model = BrainModel(config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    target = CANDIDATE if args.candidate else ACTIVE; target.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and target.exists():
        previous = torch.load(target, map_location=DEVICE, weights_only=False)
        if previous.get("config") == config.__dict__:
            model.load_state_dict(previous["model_state"], strict=True); print(f"Resumed compatible checkpoint: {target}")
        else: print("Checkpoint architecture differs; starting a fresh RB3 run.")
    total_steps = supervised_tokens = 0
    for epoch in range(1, args.epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); train_losses = []; partial = False
        for step, (x, y, weights) in enumerate(train_loader, 1):
            x, y, weights = x.to(DEVICE), y.to(DEVICE), weights.to(DEVICE)
            logits, _ = model(x, None); loss = weighted_loss(logits, y, weights); (loss / max(1, args.grad_accum)).backward()
            supervised_tokens += int((y >= 0).sum().item())
            if step % max(1, args.grad_accum) == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); optimizer.zero_grad(set_to_none=True)
            train_losses.append(loss.item()); total_steps += 1
            if args.max_steps and step >= args.max_steps: partial = step < len(train_loader); break
        if train_losses and len(train_losses) % max(1, args.grad_accum):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        validation_loss, validation_steps = evaluate(model, validation_loader, args.max_val_steps)
        metadata = {"epochs_requested": args.epochs, "epochs_completed": epoch if not partial else 0, "partial_epoch": partial, "training_steps_this_run": total_steps, "supervised_tokens_this_run": supervised_tokens, "train_loss": sum(train_losses) / max(1, len(train_losses)), "val_loss": validation_loss, "validation_signature": heldout_signature, "validation_batches": validation_steps, "validation_sampled": bool(args.max_val_steps), "device": DEVICE, "run_label": args.run_label}
        torch.save(checkpoint_payload(model, config, len(dataset), metadata), target)
        if not args.candidate: torch.save(optimizer.state_dict(), OPTIMIZER)
        print(f"epoch={epoch} train={metadata['train_loss']:.4f} val={validation_loss:.4f} partial={partial} saved={target}")


if __name__ == "__main__": main()
