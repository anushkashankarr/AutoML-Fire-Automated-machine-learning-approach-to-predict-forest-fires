from imports import *
from config.config import CFG

class FireDL(nn.Module):
    def __init__(self, model_type, input_dim, hidden_dim, dropout):
        super().__init__()

        self.model_type = model_type

        if model_type == "lstm":
            self.seq = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            core_out_dim = hidden_dim

        elif model_type == "gru":
            self.seq = nn.GRU(input_dim, hidden_dim, batch_first=True)
            core_out_dim = hidden_dim

        else:  # transformer encoder
            n_heads = 2
            d_model = input_dim

            # Ensure divisible dimension
            if d_model % n_heads != 0:
                d_model = n_heads * ((d_model // n_heads) + 1)
                self.proj = nn.Linear(input_dim, d_model)
            else:
                self.proj = None

            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dropout=dropout,
                batch_first=False  # transformer default
            )
            self.seq = nn.TransformerEncoder(layer, num_layers=2)
            core_out_dim = d_model

        self.head = nn.Sequential(
            nn.Linear(core_out_dim, core_out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(core_out_dim // 2, 1),
        )

    def forward(self, x):
        if self.model_type in {"lstm", "gru"}:
            out, _ = self.seq(x)  # (batch, seq, hidden)
        else:
            # apply embedding projection if created
            if hasattr(self, "proj") and self.proj is not None:
                x = self.proj(x)

            # transformer expects (seq_len, batch, dim)
            x = x.permute(1, 0, 2)

            out = self.seq(x)  # (seq, batch, dim)

            # back to (batch, seq, dim)
            out = out.permute(1, 0, 2)

        z = out[:, -1, :]  # last timestep
        return torch.sigmoid(self.head(z))

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        logits: sigmoid outputs
        targets: smoothed / true labels
        """

        bce = F.binary_cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-bce)

        return ((1-pt) ** self.gamma * bce).mean()

class DLClassifierWrapper:
    """
    Sklearn-style wrapper around FireDL so that:
    - predict_proba(X) available
    - registry + predict_zone code same reh sakta hai
    """
    def __init__(self, dl_model, seq_len, device="cpu"):
        self.model = dl_model.to(device)
        self.seq_len = seq_len
        self.device = device

    def _prepare_sequences(self, X):
        if isinstance(X, pd.DataFrame):
            X_np = X.values
        else:
            X_np = np.asarray(X)
        Xs = make_sequence(X_np, self.seq_len)
        Xs = torch.tensor(Xs, dtype=torch.float32).to(self.device)
        return Xs

    def predict_proba(self, X):
        self.model.eval()
        Xs = self._prepare_sequences(X)
        with torch.no_grad():
            probs = self.model(Xs).cpu().numpy().flatten()

        # numeric safety
        probs = np.clip(probs, 1e-6, 1 - 1e-6)
        proba_2d = np.vstack([1.0 - probs, probs]).T   # shape (n, 2)
        return proba_2d

def prepare_loaded_model(model):
    """
    Ensures DL models load safely:
    - moved to CPU
    - switched to eval mode
    """
    try:
        # If it's DL wrapper, it has .model inside
        if hasattr(model, "model") and hasattr(model.model, "eval"):
            model.model.to("cpu")
            model.model.eval()
    except:
        pass

    return model

def __init__(self, model_type, input_dim, hidden_dim, dropout):
  super().__init__()

  self.model_type = model_type

  if model_type == "lstm":
      self.seq = nn.LSTM(input_dim, hidden_dim, batch_first=True)
  elif model_type == "gru":
      self.seq = nn.GRU(input_dim, hidden_dim, batch_first=True)
  else:
      layer = nn.TransformerEncoderLayer(
          d_model=input_dim, nhead=2, dropout=dropout
      )
      self.seq = nn.TransformerEncoder(layer, num_layers=2)

  self.head = nn.Sequential(
      nn.Linear(hidden_dim, hidden_dim // 2),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(hidden_dim // 2, 1),
  )

def forward(self, x):
    if self.model_type in {"lstm", "gru"}:
        out, _ = self.seq(x)
    else:
        out = self.seq(x)

    z = out[:, -1, :]  # last timestep embedding
    return torch.sigmoid(self.head(z))

def make_sequence(X, seq_len):
    n = len(X)
    if n < seq_len: 
        seq_len = n
    xs = []
    for i in range(n):
        start = max(0, i-seq_len+1)
        win = X[start:i+1]
        if len(win) < seq_len:
            pad = np.zeros((seq_len-len(win), X.shape[1]))
            win = np.vstack((pad, win))
        xs.append(win)
    return np.stack(xs)
