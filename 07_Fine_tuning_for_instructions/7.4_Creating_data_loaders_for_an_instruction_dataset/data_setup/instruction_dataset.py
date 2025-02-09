from data_setup.data_preprocessing import train_data, val_data, test_data

import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
from functools import partial


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))
    
    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


def format_input(entry):
    # Alpaca-style prompt formatting
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device="cpu"):
    
    # Find the longest sequence in the batch
    batch_max_length = max(len(item) +1 for item in batch)

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()

        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        
        # Pad sequences to max_length
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )

        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets

        # New: Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # New: Optionally truncate to maximum sequence length
        # This is useful if datasets are used that exceed the
        # 1,024-token context size supported by the GPT-2 model.
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs and targets to tensors and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


tokenizer = tiktoken.get_encoding("gpt2")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


#            Instantiating Data Loaders

# Use partial to fix 'device' and 'allowed_max_length'.  
custom_collate_fn = partial(
    custom_collate_fn,
    device=device,
    allowed_max_length=1024
)

num_workers = 0
batch_size = 8

torch.manual_seed(123)

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=custom_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=custom_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=custom_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

# print("Train loader:")
# for inputs, targets in train_loader:
    # print(inputs.shape, targets.shape)

# First input:
# torch.Size([8, 61]) torch.Size([8, 61])
# 8 represents the batch size and 61 is the number 
# of tokens in each training example in this batch.

# Checking to see if the inputs contain the 
# <|endoftext|> padding token corresponding to token ID 50256

# print(inputs[0])
# Print:
# tensor([21106,   318,   281, 12064,   326,  8477,   257,  4876,    13, 19430,
#           257,  2882,   326, 20431, 32543,   262,  2581,    13,   198,   198,
#         21017, 46486,    25,   198, 30003,  6525,   262,  6827,  1262,   257,
#           985,   576,    13,   198,   198, 21017, 23412,    25,   198,   464,
#          5156,   318,   845, 13779,    13,   198,   198, 21017, 18261,    25,
#           198,   464,  5156,   318,   355, 13779,   355,   257,  4936,    13,
#         50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256])

# Similarly, we visually double-check that the targets contain the -100 placeholder tokens
# print(targets[0])
# Print:
# tensor([  318,   281, 12064,   326,  8477,   257,  4876,    13, 19430,   257,
#          2882,   326, 20431, 32543,   262,  2581,    13,   198,   198, 21017,
#         46486,    25,   198, 30003,  6525,   262,  6827,  1262,   257,   985,
#           576,    13,   198,   198, 21017, 23412,    25,   198,   464,  5156,
#           318,   845, 13779,    13,   198,   198, 21017, 18261,    25,   198,
#           464,  5156,   318,   355, 13779,   355,   257,  4936,    13, 50256,
#          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100])