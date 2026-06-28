"""
Train the PRISM soft gating module (Eq. 4-5 in the paper).

The gating module is a light-weight classifier f_theta that maps the sensitivity
profiling features z = [R(P), d] (risk score + entity sensitivity mask) to a
distribution pi over the three execution paths (cloud / collaborative / edge).
Following the paper, it is trained with a task loss plus an entropy penalty:

    L_gating = L_task + lambda * H(pi)               (Eq. 5)

Features are produced by the real edge-side profiling module
(``edge_detection.EdgeEntityDetector``) and the same ``prepare_features`` routine
used at inference, so there is no train/inference skew. Supervision comes from
``Dataset/routing_dataset.xlsx`` (see ``generate_training_data.py``).

Usage:
    python train_soft_gating.py
"""

import os
import logging

import numpy as np
import pandas as pd
import torch

from edge_detection import EdgeEntityDetector
from soft_gating import SoftGatingModule, SoftGatingPredictor, RoutingMode

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("train_soft_gating")
logging.getLogger("EdgeEntityDetector").setLevel(logging.WARNING)
logging.getLogger("presidio-analyzer").setLevel(logging.WARNING)

SEED = 42
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "Dataset", "routing_dataset.xlsx")
SAVE_PATH = os.path.join(os.path.dirname(__file__), "models", "soft_gating_pretrained.pth")

# Route label -> RoutingMode index (matches RoutingMode enum / softmax order).
ROUTE_TO_IDX = {
    "edge": RoutingMode.EDGE_ONLY.value,
    "collaborative": RoutingMode.COLLABORATIVE.value,
    "cloud": RoutingMode.CLOUD_ONLY.value,
}
IDX_TO_ROUTE = {v: k for k, v in ROUTE_TO_IDX.items()}

EPOCHS = 200
LR = 1e-2
LAMBDA_ENTROPY = 0.4
VAL_FRACTION = 0.25


def build_features(prompts):
    """Run edge-side profiling and the inference-time feature prep for each prompt."""
    detector = EdgeEntityDetector()
    predictor = SoftGatingPredictor(model_path=None)  # only used for prepare_features
    feats = []
    for p in prompts:
        res = detector.detect_and_classify(p)
        z = predictor.prepare_features(res["risk_score"], res["sensitivity_labels"])
        feats.append(z.squeeze(0))
    return torch.stack(feats)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    df = pd.read_excel(DATA_PATH)
    logger.info(f"Loaded {len(df)} labelled prompts from {os.path.normpath(DATA_PATH)}")

    logger.info("Extracting profiling features (NER + risk scoring)...")
    X = build_features(df["prompt"].tolist())
    y = torch.tensor([ROUTE_TO_IDX[r] for r in df["route"]], dtype=torch.long)

    # Stratified-ish split via a fixed shuffle.
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(y), generator=g)
    n_val = int(len(y) * VAL_FRACTION)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    Xtr, ytr = X[train_idx], y[train_idx]
    Xva, yva = X[val_idx], y[val_idx]

    model = SoftGatingModule()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc, best_state = 0.0, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        pi = model(Xtr)
        loss, parts = model.compute_loss(pi, ytr, lambda_entropy=LAMBDA_ENTROPY)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            tr_acc = (model(Xtr).argmax(-1) == ytr).float().mean().item()
            val_pi = model(Xva)
            val_acc = (val_pi.argmax(-1) == yva).float().mean().item()
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 20 == 0 or epoch == 1:
            logger.info(f"epoch {epoch:3d}  loss={parts['total']:.4f} "
                        f"task={parts['task']:.4f} H={parts['avg_entropy']:.4f}  "
                        f"train_acc={tr_acc:.3f} val_acc={val_acc:.3f}")

    model.load_state_dict(best_state)

    # Final metrics on the full set (measured, not assumed).
    model.eval()
    with torch.no_grad():
        full_pred = model(X).argmax(-1)
    full_acc = (full_pred == y).float().mean().item()
    dist = {IDX_TO_ROUTE[i]: float((full_pred == i).float().mean()) for i in range(3)}
    logger.info(f"\nBest val accuracy: {best_val_acc:.3f}")
    logger.info(f"Full-set accuracy: {full_acc:.3f}")
    logger.info(f"Predicted routing distribution: {dist}")

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": EPOCHS,
        "metrics": {
            "best_val_acc": best_val_acc,
            "full_acc": full_acc,
            "routing_distribution": dist,
        },
        "config": {
            "input_dim": 11,
            "hidden_dim": 64,
            "dropout_rate": 0.1,
            "lambda_entropy": LAMBDA_ENTROPY,
        },
        "training_info": {
            "dataset": os.path.basename(DATA_PATH),
            "dataset_size": int(len(y)),
            "val_fraction": VAL_FRACTION,
            "learning_rate": LR,
            "optimizer": "Adam",
            "seed": SEED,
        },
    }
    torch.save(checkpoint, SAVE_PATH)
    logger.info(f"Saved trained gating checkpoint to {os.path.normpath(SAVE_PATH)}")


if __name__ == "__main__":
    main()
