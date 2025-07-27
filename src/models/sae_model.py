import torch
import torch.nn as nn
import torch.optim as optim

# Import DEVICE if defined in another file, or define it here
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


class SAE(nn.Module):
    def __init__(self, input_dim, width_ratio=4, activation=nn.ReLU()):
        super().__init__()
        self.sae_hidden = input_dim * width_ratio
        self.W_in = nn.Parameter(
            nn.init.kaiming_uniform_(
                torch.empty(input_dim, self.sae_hidden, device=DEVICE),
                nonlinearity="relu",
            )
        )
        self.b_in = nn.Parameter(torch.zeros(self.sae_hidden, device=DEVICE))
        self.W_out = nn.Parameter(
            nn.init.kaiming_uniform_(
                torch.empty(self.sae_hidden, input_dim, device=DEVICE),
                nonlinearity="relu",
            )
        )
        self.b_out = nn.Parameter(torch.zeros(input_dim, device=DEVICE))
        self.nonlinearity = activation

    def _normalize_weights(self):
        with torch.no_grad():
            norms = self.W_out.norm(p=2, dim=0, keepdim=True)
            self.W_out.div(norms)

    def forward(self, x):
        x = x - self.b_out
        acts = self.nonlinearity(x @ self.W_in + self.b_in)
        l1_regularization = acts.abs().sum()
        l0 = (acts > 0).sum(dim=1).float().mean()
        self._normalize_weights()

        return l0, l1_regularization, acts @ self.W_out + self.b_out

    def train(
        self,
        train_data,
        test_data,
        batch_size=128,
        n_epochs=1000,
        l1_lam=3e-5,
        weight_decay=1e-4,
        output_epoch=False,
    ):
        optimizer = optim.Adam(self.parameters(), weight_decay=weight_decay)
        mse_criterion = nn.MSELoss().to(DEVICE)

        n_batches = len(train_data) // batch_size
        n_test_batches = len(test_data) // batch_size

        for epoch in range(n_epochs):
            total_loss = 0
            total_test_loss = 0
            total_mse_loss = 0
            total_l1_loss = 0
            total_l0 = 0
            batch_perm = torch.randperm(len(train_data), device=DEVICE)
            test_batch_perm = torch.randperm(len(test_data), device=DEVICE)

            for i in range(n_batches):
                # training
                idx = batch_perm[i * batch_size : (i + 1) * batch_size]
                batch = train_data[idx]

                optimizer.zero_grad()
                l0, l1, recon_hiddens = self(batch)

                recon_loss = mse_criterion(recon_hiddens, batch)
                sparsity_loss = l1_lam * l1
                loss = recon_loss + sparsity_loss

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_l1_loss += sparsity_loss.item()
                total_mse_loss += recon_loss.item()
                total_l0 += l0

                # testing
                if i < n_test_batches:
                    test_idx = test_batch_perm[i * batch_size : (i + 1) * batch_size]
                    test_batch = test_data[test_idx]

                    with torch.no_grad():
                        _, _, test_recon = self(test_batch)
                        test_loss = mse_criterion(test_recon, test_batch)
                        total_test_loss += test_loss.item()

            if (epoch % 10 == 0) and (output_epoch == True):
                avg_loss = total_loss / n_batches
                avg_test_loss = total_test_loss / n_test_batches
                avg_l1_loss = total_l1_loss / n_batches
                avg_l0 = total_l0 / n_batches

                print(
                    f"Epoch {epoch}, Loss: {avg_loss:.4f}, "
                    f"Test Loss: {avg_test_loss:.4f}, "
                    f"L1: {avg_l1_loss:.4f}, "
                    f"L0: {avg_l0:.4f}"
                )

        return {
            "mse": total_mse_loss / n_batches,
            "L0": total_l0 / n_batches,
            "L1 lambda": l1_lam,
        }


class BatchedSAE(nn.Module):
    def __init__(self, input_dim, n_models, width_ratio=4, activation=nn.ReLU()):
        super().__init__()
        self.n_models = n_models
        self.sae_hidden = input_dim * width_ratio

        # Shape: [n_models, input_dim, sae_hidden]
        self.W_e = nn.Parameter(
            nn.init.xavier_normal_(
                torch.empty(n_models, input_dim, self.sae_hidden, device=DEVICE)
            )
        )
        # Shape: [n_models, sae_hidden]
        self.b_e = nn.Parameter(torch.zeros(n_models, self.sae_hidden, device=DEVICE))

        # Shape: [n_models, sae_hidden, input_dim]
        self.W_d = nn.Parameter(
            nn.init.xavier_normal_(
                torch.empty(n_models, self.sae_hidden, input_dim, device=DEVICE)
            )
        )
        # Shape: [n_models, input_dim]
        self.b_d = nn.Parameter(torch.zeros(n_models, input_dim, device=DEVICE))
        self.nonlinearity = activation

    def _normalize_weights(self):
        with torch.no_grad():
            # Normalize each model's weights separately
            # Shape: [n_models, 1, input_dim]
            norms = self.W_d.norm(p=2, dim=1, keepdim=True)
            self.W_d.div_(norms)

    def forward(self, x):
        # x shape is already: [n_models, batch_size, input_dim]
        # Subtract bias for each model
        x = x - self.b_d.unsqueeze(1)

        # Compute activations for each model
        # bmm for batched matrix multiply
        acts = self.nonlinearity(torch.bmm(x, self.W_e) + self.b_e.unsqueeze(1))

        # Calculate regularization terms for each model
        l1_regularization = acts.abs().sum(dim=[1, 2])  # [n_models]
        l0 = (acts > 0).sum(dim=2).float().mean(dim=1)  # [n_models]

        self._normalize_weights()

        # Reconstruct input for each model
        reconstruction = torch.bmm(acts, self.W_d) + self.b_out.unsqueeze(1)

        return l0, l1_regularization, reconstruction

    def train(
        self,
        train_data,
        test_data,
        batch_size=128,
        n_epochs=10000,
        l1_lam=3e-5,
        weight_decay=1e-4,
        output_epoch=False,
        patience=10,
        min_improvement=1e-4,
    ):
        # Ensure train_data shape is [n_models, n_samples, input_dim]
        assert (
            train_data.dim() == 3 and train_data.size(0) == self.n_models
        ), f"Expected train_data shape [n_models, n_samples, input_dim], got {train_data.shape}"

        n_samples = train_data.size(1)  # Use size(1) to get number of samples
        batch_size = min(batch_size, n_samples)  # Ensure batch_size <= n_samples
        n_batches = n_samples // batch_size

        # Initialize tracking variables
        optimizer = torch.optim.Adam(
            self.parameters(), lr=1e-3, weight_decay=weight_decay
        )

        # Initialize early stopping trackers for each model
        best_losses = torch.full(
            (self.n_models,), float("inf"), device=train_data.device
        )
        patience_counters = torch.zeros(self.n_models, dtype=torch.int)
        active_models = torch.ones(
            self.n_models, dtype=torch.bool, device=train_data.device
        )

        for epoch in range(n_epochs):
            # Skip iteration if all models have converged
            if not active_models.any():
                break

            indices = torch.randperm(n_samples)
            total_mse_loss = torch.zeros(self.n_models, device=train_data.device)
            total_l1_loss = torch.zeros(self.n_models, device=train_data.device)
            total_l0 = torch.zeros(self.n_models, device=train_data.device)

            # Process batches
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i : i + batch_size]
                batch = train_data[:, batch_indices]  # Select samples for all models

                optimizer.zero_grad()
                l0, l1, recon_hiddens = self(batch)

                # Calculate loss for each model separately - simplified
                recon_loss = nn.MSELoss(reduction="none")(recon_hiddens, batch).mean(
                    dim=[1, 2]
                )  # [n_models]

                sparsity_loss = l1_lam * l1
                loss = recon_loss + sparsity_loss

                # Sum losses across all models for backward
                loss.sum().backward()
                optimizer.step()

                total_mse_loss += recon_loss.detach()
                total_l1_loss += sparsity_loss.detach()
                total_l0 += l0.detach()

            # Early stopping check for each model
            avg_loss = total_mse_loss / n_batches

            for m in range(self.n_models):
                if not active_models[m]:
                    continue

                if avg_loss[m] < best_losses[m] - min_improvement:
                    best_losses[m] = avg_loss[m]
                    patience_counters[m] = 0
                else:
                    patience_counters[m] += 1
                    if patience_counters[m] >= patience:
                        active_models[m] = False
                        if output_epoch:
                            print(
                                f"Model {m} converged at epoch {epoch} with loss {best_losses[m]:.4f}"
                            )

            if (epoch % 10 == 0) and output_epoch:
                avg_l1_loss = total_l1_loss / n_batches
                avg_l0 = total_l0 / n_batches

                for m in range(self.n_models):
                    if active_models[m]:
                        print(
                            f"Model {m}, Epoch {epoch}, Loss: {avg_loss[m]:.4f}, "
                            f"L1: {avg_l1_loss[m]:.4f}, "
                            f"L0: {avg_l0[m]:.4f}"
                        )

        return [
            {
                "mse": total_mse_loss[m].item() / n_batches,
                "L0": total_l0[m].item() / n_batches,
                "L1 lambda": l1_lam,
                "weights": self.W_d[m, :, :].detach().cpu().numpy(),
                "biases": self.b_d[m, :].detach().cpu().numpy(),
            }
            for m in range(self.n_models)
        ]


class BatchedSAE_Updated(nn.Module):
    """
    This is a modified version of BatchedSAE that uses a different initialization method.
    Link to the updates to SAE: https://transformer-circuits.pub/2024/april-update/index.html#training-saes

    Key changes:
    - Decoder weights (W_d) are no longer L2 normalized to unit norm
    - L1 regularization is weighted by the L2 norm of decoder columns
    - The decoder weights (W_d) are initialized with random directions and fixed L2 norm
    - The encoder weights (W_e) are initialized as the transpose of W_d
    - The decoder biases (b_d) are initialized to zero
    - The encoder biases (b_e) are initialized to zero
    """

    def __init__(self, input_dim, n_models, width_ratio=4, activation=nn.ReLU()):
        super().__init__()
        self.n_models = n_models
        self.sae_hidden = input_dim * width_ratio

        # Initialize W_d with random directions and fixed L2 norm
        W_d_init = torch.randn(n_models, self.sae_hidden, input_dim, device=DEVICE)
        # Normalize columns to have L2 norm of 0.1
        W_d_init = 0.1 * W_d_init / W_d_init.norm(p=2, dim=2, keepdim=True)

        # Initialize W_e as W_d transpose
        W_e_init = W_d_init.transpose(1, 2)

        # Shape: [n_models, input_dim, sae_hidden]
        self.W_e = nn.Parameter(W_e_init)

        # Shape: [n_models, sae_hidden]
        self.b_e = nn.Parameter(torch.zeros(n_models, self.sae_hidden, device=DEVICE))

        # Shape: [n_models, sae_hidden, input_dim]
        self.W_d = nn.Parameter(W_d_init)

        # Shape: [n_models, input_dim]
        self.b_d = nn.Parameter(torch.zeros(n_models, input_dim, device=DEVICE))
        self.nonlinearity = activation

    def forward(self, x):
        # x shape is already: [n_models, batch_size, input_dim]

        # Compute activations f(x) for each model
        # bmm for batched matrix multiply
        acts = self.nonlinearity(
            torch.bmm(x, self.W_e)
            + self.b_e.unsqueeze(1)  # [n_models, batch_size, sae_hidden]
        )

        # Calculate L1 regularization weighted by decoder norm for each feature
        # [n_models, batch_size, sae_hidden] * [n_models, sae_hidden, 1] -> [n_models]
        decoder_norms = self.W_d.norm(p=2, dim=2)  # [n_models, sae_hidden]
        l1_regularization = (acts.abs() * decoder_norms.unsqueeze(1)).sum(
            dim=[1, 2]
        )  # [n_models]

        # Calculate L0 sparsity metric
        l0 = (acts > 0).sum(dim=2).float().mean(dim=1)  # [n_models]

        # Reconstruct input for each model
        reconstruction = torch.bmm(acts, self.W_d) + self.b_d.unsqueeze(
            1
        )  # [n_models, batch_size, input_dim]

        return l0, l1_regularization, reconstruction

    def train(
        self,
        train_data,
        test_data,
        batch_size=128,
        n_epochs=10000,
        l1_lam=3e-5,
        weight_decay=1e-4,
        output_epoch=False,
        patience=10,
        min_improvement=1e-4,
    ):
        # Ensure train_data shape is [n_models, n_samples, input_dim]
        assert (
            train_data.dim() == 3 and train_data.size(0) == self.n_models
        ), f"Expected train_data shape [n_models, n_samples, input_dim], got {train_data.shape}"

        n_samples = train_data.size(1)  # Use size(1) to get number of samples
        batch_size = min(batch_size, n_samples)  # Ensure batch_size <= n_samples
        n_batches = n_samples // batch_size

        # Initialize tracking variables
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)

        # Initialize early stopping trackers for each model
        best_losses = torch.full(
            (self.n_models,), float("inf"), device=train_data.device
        )
        patience_counters = torch.zeros(self.n_models, dtype=torch.int)
        active_models = torch.ones(
            self.n_models, dtype=torch.bool, device=train_data.device
        )

        for epoch in range(n_epochs):
            # Skip iteration if all models have converged
            if not active_models.any():
                break

            indices = torch.randperm(n_samples)
            total_mse_loss = torch.zeros(self.n_models, device=train_data.device)
            total_l1_loss = torch.zeros(self.n_models, device=train_data.device)
            total_l0 = torch.zeros(self.n_models, device=train_data.device)

            # Process batches
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i : i + batch_size]
                batch = train_data[:, batch_indices]  # Select samples for all models

                optimizer.zero_grad()
                l0, l1, recon_hiddens = self(batch)

                # Calculate loss for each model separately - simplified
                recon_loss = nn.MSELoss(reduction="none")(recon_hiddens, batch).mean(
                    dim=[1, 2]
                )  # [n_models]

                sparsity_loss = l1_lam * l1
                loss = recon_loss + sparsity_loss

                # Sum losses across all models for backward
                loss.sum().backward()
                optimizer.step()

                total_mse_loss += recon_loss.detach()
                total_l1_loss += sparsity_loss.detach()
                total_l0 += l0.detach()

            # Early stopping check for each model
            avg_loss = total_mse_loss / n_batches

            for m in range(self.n_models):
                if not active_models[m]:
                    continue

                if avg_loss[m] < best_losses[m] - min_improvement:
                    best_losses[m] = avg_loss[m]
                    patience_counters[m] = 0
                else:
                    patience_counters[m] += 1
                    if patience_counters[m] >= patience:
                        active_models[m] = False
                        if output_epoch:
                            print(
                                f"Model {m} converged at epoch {epoch} with loss {best_losses[m]:.4f}"
                            )

            if (epoch % 10 == 0) and output_epoch:
                avg_l1_loss = total_l1_loss / n_batches
                avg_l0 = total_l0 / n_batches

                for m in range(self.n_models):
                    if active_models[m]:
                        print(
                            f"Model {m}, Epoch {epoch}, Loss: {avg_loss[m]:.4f}, "
                            f"L1: {avg_l1_loss[m]:.4f}, "
                            f"L0: {avg_l0[m]:.4f}"
                        )

        return [
            {
                "mse": total_mse_loss[m].item() / n_batches,
                "L0": total_l0[m].item() / n_batches,
                "L1 lambda": l1_lam,
                "weights": self.W_d[m, :, :].detach().cpu().numpy(),
                "biases": self.b_d[m, :].detach().cpu().numpy(),
            }
            for m in range(self.n_models)
        ]
