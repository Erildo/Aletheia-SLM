"""Training utilities for HyperFocus models"""

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def train_epoch(model, train_loader, optimizer, device, epoch, total_epochs):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    losses = []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs}")
    
    for batch_idx, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        
        # Forward pass
        logits, routing_scores = model(input_ids)
        
        # Language modeling loss
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            input_ids[:, 1:].reshape(-1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        losses.append(loss.item())
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg': f'{total_loss/(batch_idx+1):.4f}'
        })
    
    avg_loss = total_loss / len(train_loader)
    return losses, avg_loss


def train_model(model, train_loader, epochs, lr, device, model_name="model"):
    """Complete training loop"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    all_losses = []
    
    for epoch in range(1, epochs + 1):
        epoch_losses, avg_loss = train_epoch(
            model, train_loader, optimizer, device, epoch, epochs
        )
        all_losses.extend(epoch_losses)
        
        print(f"\n{model_name} - Epoch {epoch} Average Loss: {avg_loss:.4f}")
    
    return all_losses


@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_length=50, temperature=0.8, device='cpu'):
    """Generate text from a prompt"""
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    for _ in range(max_length):
        logits, _ = model(input_ids)
        next_token_logits = logits[:, -1, :] / temperature
        next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)
