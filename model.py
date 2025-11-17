import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ExpandCompressMLP(nn.Module):
    def __init__(
        self,
        input_dim=1024,
        hidden_dim=768,
        bottleneck_dim=384,
        output_dim=1536,
        dropout=0.1,
        use_residual=True
    ):
        super().__init__()
        self.use_residual = use_residual
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # Residual connection 
        if use_residual:
            self.residual = nn.Linear(input_dim, output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        h = self.encoder(x)
        out = self.decoder(h)
        
        if self.use_residual:
            out = out + self.residual(x)
        
        return out


def info_nce_loss(outputs, targets, temperature=0.05):

    # Compute logits via dot product
    logits = outputs @ targets.T 
    logits = logits / temperature

    # Correct labels 
    labels = torch.arange(logits.size(0), device=outputs.device)

    # Compute cross-entropy loss
    loss = F.cross_entropy(logits, labels)
    return loss



def train_model(
    model, 
    train_loader, 
    val_loader, 
    device='cuda',
    epochs=50,
    lr=1e-3,
    weight_decay=1e-4,
    temperature=0.05,
    patience=10,
    use_scheduler=True,
    use_amp=True,
    grad_clip=1.0
):

    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Scheduler
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    
    for epoch in range(epochs):
        # TRAINING 
        model.train()
        train_loss = 0
        
        for text_batch, img_batch in train_loader:
            text_batch = text_batch.to(device)
            img_batch = img_batch.to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(text_batch)
                loss = info_nce_loss(outputs, img_batch, temperature)
            
            # Backward
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # VALIDATION 
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for text_batch, img_batch in val_loader:
                text_batch = text_batch.to(device)
                img_batch = img_batch.to(device)
                
                outputs = model(text_batch)
                loss = info_nce_loss(outputs, img_batch, temperature)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Update scheduler
        if use_scheduler:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = lr
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train={train_loss:.6f}, Val={val_loss:.6f}, LR={current_lr:.2e}")
        
        # Early stopping
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, 'best_model.pth')
            print(f"  ✓ Best model saved (val_loss: {best_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stopping at epoch {epoch+1}')
                break
    
    # Load best model
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, history


class EnsembleModel(nn.Module):
    def __init__(self, models, weights=None):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.weights = weights

    def forward(self, x):
        preds = []

        for model in self.models:
            out = model(x)
            out = torch.nn.functional.normalize(out, p=2, dim=1)  
            preds.append(out)

        stacked = torch.stack(preds, dim=0) 

        if self.weights is None:
            return stacked.mean(dim=0)
        else:
            w = torch.tensor(self.weights, dtype=torch.float32, device=x.device).view(-1, 1, 1)
            out = (stacked * w).sum(dim=0)
            return torch.nn.functional.normalize(out, p=2, dim=1)