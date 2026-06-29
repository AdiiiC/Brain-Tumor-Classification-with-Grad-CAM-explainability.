"""
Upgrade #15 — Federated Learning.

Train across multiple hospitals without sharing raw patient data.
Each hospital trains locally and shares only model weight updates (gradients).
Critical for medical data privacy (HIPAA/GDPR compliance).

Uses Flower (flwr) framework for federated orchestration.
Install: pip install flwr
"""

import numpy as np
from pathlib import Path

try:
    import flwr as fl
    import tensorflow as tf
    FLOWER_AVAILABLE = True
except ImportError:
    FLOWER_AVAILABLE = False


# ── Federated Client ──────────────────────────────────────────────────────────

class BrainTumorClient(fl.client.NumPyClient if FLOWER_AVAILABLE else object):
    """
    Federated Learning client for a single hospital/institution.

    Each client:
    1. Receives global model weights from the server
    2. Trains on local data (never leaves the hospital)
    3. Sends only weight UPDATES back to the server
    """

    def __init__(self, model, train_data, val_data, epochs_per_round=3):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.epochs_per_round = epochs_per_round

    def get_parameters(self, config):
        """Return current model weights."""
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Train on local data and return updated weights."""
        self.model.set_weights(parameters)

        self.model.fit(
            self.train_data,
            epochs=self.epochs_per_round,
            validation_data=self.val_data,
            verbose=0,
        )

        return self.model.get_weights(), len(self.train_data), {}

    def evaluate(self, parameters, config):
        """Evaluate global model on local test data."""
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.val_data, verbose=0)
        return loss, len(self.val_data), {"accuracy": accuracy}


# ── Federated Server Strategy ─────────────────────────────────────────────────

def create_federated_strategy(min_clients: int = 2, num_rounds: int = 10):
    """
    Create a FedAvg strategy for the central server.

    FedAvg: Averages model weights from all participating hospitals,
    weighted by their dataset size.
    """
    if not FLOWER_AVAILABLE:
        raise ImportError("Flower required: pip install flwr")

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,          # Use all available clients per round
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
    )
    return strategy


def start_server(num_rounds: int = 10, min_clients: int = 2):
    """Start the federated learning server."""
    if not FLOWER_AVAILABLE:
        raise ImportError("Flower required: pip install flwr")

    strategy = create_federated_strategy(min_clients, num_rounds)

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )


def start_client(server_address: str, model, train_data, val_data):
    """Start a federated learning client (one per hospital)."""
    if not FLOWER_AVAILABLE:
        raise ImportError("Flower required: pip install flwr")

    client = BrainTumorClient(model, train_data, val_data)
    fl.client.start_numpy_client(server_address=server_address, client=client)


# ── Differential Privacy Extension ───────────────────────────────────────────

def add_differential_privacy(model, noise_multiplier=1.0, l2_norm_clip=1.0):
    """
    Add differential privacy guarantees to training.

    Clips gradients and adds calibrated noise so individual patient
    data cannot be reverse-engineered from the shared model updates.

    Requires: pip install tensorflow-privacy
    """
    try:
        from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer

        dp_optimizer = DPKerasAdamOptimizer(
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            num_microbatches=1,
            learning_rate=1e-4,
        )

        model.compile(
            optimizer=dp_optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        print(f"Differential privacy enabled (ε-δ guarantees)")
        print(f"  Noise multiplier: {noise_multiplier}")
        print(f"  L2 norm clip: {l2_norm_clip}")
        return model

    except ImportError:
        raise ImportError("tensorflow-privacy required: pip install tensorflow-privacy")
