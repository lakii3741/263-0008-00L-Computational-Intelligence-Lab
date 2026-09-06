"""Standard Neural Matrix Factorization (NeuMF) Predictor.

Implements the standard NeuMF architecture from:
He et al., "Neural Collaborative Filtering", WWW 2017.

It fuses Generalized Matrix Factorization (GMF) and a Multi-Layer 
Perceptron (MLP) to learn user-item interactions.
"""

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cil.predictors.base import Predictor


class NeuMFModule(nn.Module):
    """The PyTorch Neural Network architecture for standard NeuMF."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int = 16,
        mlp_layers: tuple[int, ...] = (64, 32, 16, 8),
        use_wishlist: bool = False
    ):
        super().__init__()
        self.use_wishlist = use_wishlist
        
        # --- PATHWAY 1: GMF ---
        self.embed_user_gmf = nn.Embedding(num_users, embed_dim)
        self.embed_item_gmf = nn.Embedding(num_items, embed_dim)

        # --- PATHWAY 2: MLP ---
        mlp_embed_dim = mlp_layers[0] // 2
        self.embed_user_mlp = nn.Embedding(num_users, mlp_embed_dim)
        self.embed_item_mlp = nn.Embedding(num_items, mlp_embed_dim)

        mlp_modules = []
        
        # If wishlist is used, the first MLP layer must accept 1 extra dimension
        in_dim = mlp_layers[0] + (1 if self.use_wishlist else 0)
        
        for idx in range(len(mlp_layers) - 1):
            mlp_modules.append(nn.Linear(in_dim, mlp_layers[idx + 1]))
            mlp_modules.append(nn.ReLU())
            in_dim = mlp_layers[idx + 1] # Subsequent layers use the standard defined sizes
            
        self.mlp = nn.Sequential(*mlp_modules)

        # --- FUSION LAYER ---
        fusion_dim = embed_dim + mlp_layers[-1]
        self.fusion = nn.Linear(fusion_dim, 1)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with standard distributions."""
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, u: torch.Tensor, i: torch.Tensor, w: torch.Tensor | None = None) -> torch.Tensor:
        # GMF Pathway
        u_gmf = self.embed_user_gmf(u)
        i_gmf = self.embed_item_gmf(i)
        gmf_out = u_gmf * i_gmf  

        # MLP Pathway
        u_mlp = self.embed_user_mlp(u)
        i_mlp = self.embed_item_mlp(i)
        
        # Inject the wishlist flag if active
        if self.use_wishlist and w is not None:
            mlp_in = torch.cat([u_mlp, i_mlp, w.unsqueeze(1)], dim=-1)
        else:
            mlp_in = torch.cat([u_mlp, i_mlp], dim=-1)
            
        mlp_out = self.mlp(mlp_in)

        # Fusion
        fused = torch.cat([gmf_out, mlp_out], dim=-1)
        out = self.fusion(fused)
        
        return self.sigmoid(out).squeeze()


class NeuMFPredictor(Predictor):
    """
    Standard NeuMF predictor compatible with the CIL pipeline.
    """

    def __init__(
        self,
        embed_dim: int = 16,
        mlp_layers: tuple[int, ...] = (64, 32, 16, 8),
        epochs: int = 20,
        batch_size: int = 256,
        lr: float = 0.001,
        seed: int = 42,
        use_wishlist: bool = False
    ) -> None:
        self._embed_dim = embed_dim
        self._mlp_layers = tuple(mlp_layers)
        self._epochs = epochs
        self._batch_size = batch_size
        self._lr = lr
        self._seed = seed
        self._use_wishlist = use_wishlist
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model: NeuMFModule | None = None
        self._wishlist_set: set[tuple[int, int]] = set()

    def fit(
        self,
        train_df: pd.DataFrame,
        num_scientists: int,
        num_papers: int,
        wishlist: pd.DataFrame | None = None,
    ) -> None:
        print(f"Building NeuMF Model (Wishlist: {self._use_wishlist}) on device: {self._device}")
        torch.manual_seed(self._seed)
        random.seed(self._seed)
        np.random.seed(self._seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
        self._model = NeuMFModule(
            num_users=num_scientists,
            num_items=num_papers,
            embed_dim=self._embed_dim,
            mlp_layers=self._mlp_layers,
            use_wishlist=self._use_wishlist
        ).to(self._device)

        users = torch.tensor(train_df['sid'].values, dtype=torch.long)
        items = torch.tensor(train_df['pid'].values, dtype=torch.long)
        ratings = torch.tensor(train_df['rating'].values / 5.0, dtype=torch.float32)

        # --- WISHLIST PROCESSING ---
        if self._use_wishlist:
            if wishlist is None:
                raise ValueError("use_wishlist is True, but no wishlist DataFrame was provided!")
            
            # Fast Pandas merge to align wishlist flags with the training ratings
            wishlist_df = wishlist.copy()
            wishlist_df['in_wishlist'] = 1.0
            merged = train_df.merge(wishlist_df, on=['sid', 'pid'], how='left')
            merged['in_wishlist'] = merged['in_wishlist'].fillna(0.0)
            
            w_tensor = torch.tensor(merged['in_wishlist'].values, dtype=torch.float32)
            dataset = TensorDataset(users, items, ratings, w_tensor)
            
            # Store the set for fast O(1) lookups during inference
            self._wishlist_set = set(zip(wishlist['sid'], wishlist['pid']))
        else:
            dataset = TensorDataset(users, items, ratings)
        # ---------------------------

        g = torch.Generator()
        g.manual_seed(self._seed)

        dataloader = DataLoader(
            dataset, 
            batch_size=self._batch_size, 
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            worker_init_fn=_seed_worker,
            generator=g
        )

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr)
        criterion = nn.BCELoss()

        self._model.train()
        for epoch in range(self._epochs):
            total_loss = 0.0
            
            for batch in dataloader:
                optimizer.zero_grad()
                
                # Unpack dynamically based on whether wishlist is active
                if self._use_wishlist:
                    u_b, i_b, r_b, w_b = [b.to(self._device) for b in batch]
                    preds = self._model(u_b, i_b, w_b)
                else:
                    u_b, i_b, r_b = [b.to(self._device) for b in batch]
                    preds = self._model(u_b, i_b)
                
                loss = criterion(preds, r_b)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            print(f"Epoch {epoch + 1:02d}/{self._epochs} | Loss: {total_loss / len(dataloader):.4f}")

    def predict(self, sids: np.ndarray, pids: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ValueError("Model is not fitted yet.")
            
        self._model.eval()
        users_t = torch.tensor(sids, dtype=torch.long).to(self._device)
        items_t = torch.tensor(pids, dtype=torch.long).to(self._device)
        
        with torch.no_grad():
            if self._use_wishlist:
                # Reconstruct the binary mask for the Kaggle test pairs
                w_flags = [1.0 if (s, p) in self._wishlist_set else 0.0 for s, p in zip(sids, pids)]
                w_tensor = torch.tensor(w_flags, dtype=torch.float32).to(self._device)
                preds = self._model(users_t, items_t, w_tensor)
            else:
                preds = self._model(users_t, items_t)
            
        preds_scaled = preds.cpu().numpy() * 5.0
        return np.clip(preds_scaled, 1.0, 5.0)

def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)