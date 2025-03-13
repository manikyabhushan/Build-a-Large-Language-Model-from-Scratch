import torch

# 3D vector dimensions - input
inputs = torch.tensor([
    [0.43, 0.15, 0.89], # Your      (x^1)
    [0.55, 0.87, 0.66], # journey   (x^2)
    [0.57, 0.85, 0.64], # starts    (x^3)
    [0.22, 0.58, 0.33], # with      (x^4)
    [0.77, 0.25, 0.10], # one       (x^5)
    [0.05, 0.80, 0.55]  # step      (x^6)
])

query = inputs[1] # journey   (x^2)
attn_scores_2 = torch.empty(inputs.shape[0]) # shape[0] = 6

# dot product vector - |a||b|cos θ -> cos 0 degree is 1 -> aliged to one another
for i, x_i, in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)

print("\nAttention Scores 2:\n", attn_scores_2, "\n")

# Understanding dot product
print("** Elements being multiplied in for loop **")
res = 0
for idx, element in enumerate(inputs[0]):
    print(inputs[0][idx], "*", query[idx])
    res += inputs[0][idx] * query[idx]
print("\nDot Product Manual Example:", res)
print("Dot Product Torch Example:", torch.dot(inputs[0], query))

# Normalization
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("\nAttention Weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())

# Softmax Normalization
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)
print("\nAttention Weights Naive:", attn_weights_2_naive)
print("Sum Naive:", attn_weights_2_naive.sum())

# PyTorch Softmax Normalization (better)
attn_weights_2_torch = torch.softmax(attn_scores_2, dim=0)
print("\nAttention Weights PyTorch:", attn_weights_2_torch)
print("Sum PyTorch:", attn_weights_2_torch.sum())

# Calculating the context vector for the 2nd input (journey (x^2))
query = inputs[1]
context_vec_2 = torch.zeros(query.shape) # shape = 3
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2_torch[i] * x_i

print("\nContext Vector 2:", context_vec_2)

###########################################################
# two for loops make it slow, if we need to calculate the dot product vector for all inputs
attn_scores = torch.empty(6, 6)

for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)

print(attn_scores)

# other way to get dot product vector for all inputs
attn_scores = inputs @ inputs.T
print(attn_scores)

# normalization for all 
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)

row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Row 2 sum:", row_2_sum)
print("All row sums:", attn_weights.sum(dim=-1)) 
# dim -1 to instruct the softmax function to apply normalization along the last dimension

# all context vectors 
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)

