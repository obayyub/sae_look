import torch
import torch.nn as nn
import torch.optim as optim

# Import DEVICE if defined in another file, or define it here
DEVICE = torch.device("cpu" if torch.backends.mps.is_available() else "cpu")

class SAE(nn.Module):
    def __init__(self, input_dim, width_ratio=4, activation=nn.ReLU()):
        super().__init__()
        self.sae_hidden = input_dim * width_ratio
        self.W_in = nn.Parameter(
            nn.init.kaiming_uniform_(
                torch.empty(input_dim, self.sae_hidden, device=DEVICE), 
                nonlinearity="relu"
            )
        )
        self.b_in = nn.Parameter(torch.zeros(self.sae_hidden, device=DEVICE))
        self.W_out = nn.Parameter(
            nn.init.kaiming_uniform_(
                torch.empty(self.sae_hidden, input_dim, device=DEVICE), 
                nonlinearity="relu"
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

        return l0, l1_regularization, acts@self.W_out + self.b_out

    def train(self, train_data, test_data, batch_size=128, n_epochs=1000, 
              l1_lam=3e-5, weight_decay=1e-4, output_epoch=False):
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
                idx = batch_perm[i*batch_size: (i+1)*batch_size]
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
                    test_idx = test_batch_perm[i*batch_size: (i+1)*batch_size]
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
                
                print(f'Epoch {epoch}, Loss: {avg_loss:.4f}, '
                    f'Test Loss: {avg_test_loss:.4f}, '
                    f'L1: {avg_l1_loss:.4f}, '
                    f'L0: {avg_l0:.4f}')

        return {
            'mse': total_mse_loss/n_batches,
            'L0': total_l0/n_batches,
            'L1 lambda': l1_lam,
        }