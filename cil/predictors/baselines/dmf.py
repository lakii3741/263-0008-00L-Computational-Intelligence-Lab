"""Deep Matrix Factorization (DMF) Predictor.

Implements the standard DMF architecture from:
Xue et al., "Deep Matrix Factorization Models for Recommender Systems", IJCAI 2017.

It uses two parallel neural networks to project users and items into a 
common latent space based on their full interaction vectors, 
predicting the rating via Cosine Similarity.
"""

import random
from typing import Iterable
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cil.data_loader import DataLoader as CILDataLoader
from cil.predictors.base import Predictor


class DMFModule(nn.Module):
    """The PyTorch Neural Network architecture for Deep Matrix Factorization."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        user_matrix: torch.Tensor,
        item_matrix: torch.Tensor,
        mlp_layers: tuple[int, ...],
        in_dim_u: int,
        in_dim_i: int
    ):
        super().__init__()
        
        self.register_buffer("user_matrix", user_matrix)
        self.register_buffer("item_matrix", item_matrix)

        # --- USER PATHWAY ---
        u_layers = []
        current_dim_u = in_dim_u
        for out_dim in mlp_layers:
            u_layers.append(nn.Linear(current_dim_u, out_dim))
            u_layers.append(nn.ReLU()) 
            current_dim_u = out_dim
        self.user_mlp = nn.Sequential(*u_layers)

        # --- ITEM PATHWAY ---
        i_layers = []
        current_dim_i = in_dim_i
        for out_dim in mlp_layers:
            i_layers.append(nn.Linear(current_dim_i, out_dim))
            i_layers.append(nn.ReLU()) 
            current_dim_i = out_dim
        self.item_mlp = nn.Sequential(*i_layers)

        self._init_weights()

    def _init_weights(self):
        """Initialize linear layers using Xavier Uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, u_indices: torch.Tensor, i_indices: torch.Tensor) -> torch.Tensor:
        u_vecs = self.user_matrix[u_indices] 
        i_vecs = self.item_matrix[i_indices]  

        p = self.user_mlp(u_vecs)
        q = self.item_mlp(i_vecs)

        epsilon = 1e-8
        dot_product = (p * q).sum(dim=1)
        norm_p = p.norm(p=2, dim=1)
        norm_q = q.norm(p=2, dim=1)
        
        cos_sim = dot_product / (norm_p * norm_q + epsilon)
        return cos_sim


class DMFPredictor(Predictor):
    """Standard DMF predictor compatible with the CIL pipeline."""

    def __init__(
        self,
        mlp_layers: Iterable[int] = (64, 32, 16), 
        epochs: int = 20,
        batch_size: int = 256,
        lr: float = 0.0001,
        seed: int = 42,
        use_wishlist: bool = False
    ) -> None:
        self._mlp_layers = tuple(mlp_layers)
        self._epochs = epochs
        self._batch_size = batch_size
        self._lr = lr
        self._seed = seed
        self._use_wishlist = use_wishlist
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model: DMFModule | None = None

    def fit(
        self,
        train_df: pd.DataFrame,
        num_scientists: int,
        num_papers: int,
        wishlist: pd.DataFrame | None = None,
    ) -> None:
        print(f"Building DMF Model (Wishlist: {self._use_wishlist}) on device: {self._device}")
        
        torch.manual_seed(self._seed)
        random.seed(self._seed)
        np.random.seed(self._seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # --- CONSTRUCT DENSE RATING MATRIX ---
        dense_mat = CILDataLoader.to_matrix(train_df, num_scientists, num_papers)
        dense_mat = np.nan_to_num(dense_mat, nan=0.0)
        dense_mat = dense_mat / 5.0 

        # --- WISHLIST GLOBAL AUGMENTATION ---
        if self._use_wishlist:
            if wishlist is None:
                raise ValueError("use_wishlist is True, but no wishlist DataFrame was provided!")
            
            # Construct a dense binary matrix for the entire wishlist
            wishlist_mat = np.zeros((num_scientists, num_papers), dtype=np.float32)
            sids = wishlist['sid'].values
            pids = wishlist['pid'].values
            wishlist_mat[sids, pids] = 1.0

            # Concatenate features globally (Doubles the width of the input vectors)
            user_features = np.concatenate([dense_mat, wishlist_mat], axis=1)
            item_features = np.concatenate([dense_mat.T, wishlist_mat.T], axis=1)
            
            in_dim_u = num_papers * 2
            in_dim_i = num_scientists * 2
        else:
            user_features = dense_mat
            item_features = dense_mat.T
            in_dim_u = num_papers
            in_dim_i = num_scientists

        user_matrix_t = torch.tensor(user_features, dtype=torch.float32)
        item_matrix_t = torch.tensor(item_features, dtype=torch.float32)

        self._model = DMFModule(
            num_users=num_scientists,
            num_items=num_papers,
            user_matrix=user_matrix_t,
            item_matrix=item_matrix_t,
            mlp_layers=self._mlp_layers,
            in_dim_u=in_dim_u,
            in_dim_i=in_dim_i
        ).to(self._device)

        # --- PREPARE DATASET (Observed Only) ---
        u_idx = torch.tensor(train_df['sid'].values, dtype=torch.long)
        i_idx = torch.tensor(train_df['pid'].values, dtype=torch.long)
        targets = torch.tensor(train_df['rating'].values / 5.0, dtype=torch.float32)

        dataset = TensorDataset(u_idx, i_idx, targets)

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
            for u_batch, i_batch, t_batch in dataloader:
                u_batch = u_batch.to(self._device)
                i_batch = i_batch.to(self._device)
                t_batch = t_batch.to(self._device)

                optimizer.zero_grad()
                preds = self._model(u_batch, i_batch)
                
                loss = criterion(preds, t_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                
            print(f"Epoch {epoch + 1:02d}/{self._epochs} | Loss: {total_loss / len(dataloader):.4f}")

    def predict(self, sids: np.ndarray, pids: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ValueError("Model is not fitted yet.")
            
        self._model.eval()
        
        # Keep tensors on CPU initially
        users_t = torch.tensor(sids, dtype=torch.long)
        items_t = torch.tensor(pids, dtype=torch.long)
        
        # Create a DataLoader to batch the inference
        dataset = TensorDataset(users_t, items_t)
        
        # We can use a slightly larger batch size for inference since we don't 
        # store gradients, but keeping it at self._batch_size is 100% safe.
        # CRITICAL: shuffle=False ensures the predictions stay in the exact order 
        # required for the Kaggle submission!
        dataloader = DataLoader(
            dataset, 
            batch_size=self._batch_size * 2, 
            shuffle=False, 
            num_workers=2,
            pin_memory=True
        )
        
        all_preds = []
        
        with torch.no_grad():
            for u_batch, i_batch in dataloader:
                # Move only the current mini-batch to the GPU
                u_batch = u_batch.to(self._device)
                i_batch = i_batch.to(self._device)
                
                preds = self._model(u_batch, i_batch)
                
                # Move predictions back to CPU and store them
                all_preds.append(preds.cpu().numpy())
                
        # Concatenate all the mini-batch predictions into one flat array
        preds_scaled = np.concatenate(all_preds) * 5.0
        
        # Clip to ensure predictions match exactly the CIL environment bounds
        return np.clip(preds_scaled, 1.0, 5.0)

def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)