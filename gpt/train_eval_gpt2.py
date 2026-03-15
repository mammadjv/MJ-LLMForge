import os
import argparse
import json
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn import init as init
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torch.utils.data import RandomSampler


import gpt_dataset
from gpt import GPT
from transformer import utils


def validate(val_dataloader, language_model, criterion, device):
    language_model.eval()
    total_loss = 0
    print('')
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = language_model(inputs)
                loss = criterion(outputs.view(-1, gpt_train_dataset.vocab_len), labels.view(-1))

            total_loss += loss.item()
            print(f"\rPerforming validation: Iteration {batch_idx}, Loss {loss}", end="", flush=True)

    return total_loss/len(val_dataloader)


if __name__ == "__main__":
    seq_len = 2000
    max_warmup_steps = 500
    epochs = 20

    batch_size = 4
    base_lr = 1e-4

    embed_dim = 1024
    num_heads=16
    num_layers = 24

    lr_reduction_epoch = 3
    use_checkpointing = True

    phase = str(sys.argv[1])

    gpt_train_dataset, gpt_val_dataset = gpt_dataset.create_dataset('../dataset/openwebtext', seq_len=seq_len)
    
    train_dataloader = DataLoader(
        gpt_train_dataset,
        batch_size=batch_size,
        num_workers=batch_size,  # <-- bypass workers entirely
        shuffle=False,
        generator=torch.Generator().manual_seed(42),
        )
    
    val_dataloader = DataLoader(
        gpt_val_dataset,
        batch_size=batch_size,
        num_workers=batch_size,  # <-- bypass workers entirely
        shuffle=False,
        generator=torch.Generator().manual_seed(42),
        )

    language_model = GPT(
        dictionary_length=gpt_train_dataset.vocab_len,
        seq_len=seq_len,
        num_layers=num_layers,
        embed_dim=embed_dim,
        num_heads=num_heads,
        use_checkpointing=use_checkpointing,
    )

    if torch.cuda.device_count() > 1:
        language_model = nn.DataParallel(language_model)

    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        language_model.to(device)

    decay, no_decay = utils.get_regularization_parameters(language_model, num_layers)

    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit([
        {'params': decay, 'weight_decay': 0.1},
        {'params': no_decay, 'weight_decay': 0.0},
    ], lr=base_lr)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler()

    if phase == 'train':
        language_model.apply(utils.weights_init)
        step = 0
        best_val_loss = float('inf')
        for e in range(1, epochs):
            total_train_loss = 0

            optimizer = utils.linear_lr_update(optimizer, e, base_lr, lr_reduction_epoch)

            for batch_idx, (inputs, labels) in enumerate(train_dataloader):
                step += 1
                inputs = inputs.to(device)
                labels = labels.to(device)
        
                optimizer.zero_grad()
        
                # warm_up the optimizer if needed
                if step <= max_warmup_steps:
                    utils.optimizer_warmup(optimizer, base_lr, step,max_warmup_steps)

                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = language_model(inputs)
                    loss = loss_fn(outputs.view(-1, gpt_train_dataset.vocab_len), labels.view(-1))
        
                total_train_loss += loss.item()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                print(f"\rIteration: {batch_idx}, Loss {loss}", end="", flush=True)

            val_loss = validate(val_dataloader=val_dataloader, language_model=language_model, criterion=loss_fn, device=device)

            print(f'\nEpoch {e}: avg train loss {total_train_loss/len(train_dataloader)}, avg val loss {val_loss}', flush=True)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'model_state_dict': language_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(checkpoint, f'./best_model.pt')

            language_model.train()

    else:
        checkpoint = torch.load('./best_model.pt')
        language_model.load_state_dict(checkpoint['model_state_dict'])
        language_model.eval()

        for temperature in [0.6, 0.7, 0.8, 0.9]:
            for k in [10, 20, 30]:
                print(f'####################\nGenerating with temperature {temperature}, and k {k}\n-------------------')
                start_char = np.random.randint(11, len(chars)-1)
                result = [start_char]

                # set your desired text length
                for i in range(1000):
                    if len(result) > seq_len:
                      result_tensor = torch.tensor([result[-seq_len:]]).to(device)

                    else:
                      result_tensor = torch.tensor([result]).to(device)

                    outputs = language_model(result_tensor)
                    next_token = utils.top_k_sampling(outputs, k=k, temperature=temperature)
                    result.append(next_token.item())
    
                # or save it wherever you want
                print(f'{train_dataset.decode(result)}\n<End of poem with 1000 characters>', flush=True)
