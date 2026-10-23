# %% [markdown]
# Source notebook: Data_Types.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

# %% [markdown]
# You can download the `requirements.txt` for this course from the workspace of this lab. `File --> Open...`

# %% [markdown]
# # Lesson 2: Data Types and Sizes
# 
# In this lab, you will learn about the common data types used to store the parameters of machine learning models.
# 
# 
# The libraries are already installed in the classroom.  If you're running this notebook on your own machine, you can install the following:
# 
# ```Python
# !pip install torch==2.1.1
# ```

# %%
# Notebook cell 3


# %%
# Notebook cell 4
import torch

# %% [markdown]
# ### Integers

# %%
# Notebook cell 6
# Information of `8-bit unsigned integer`
torch.iinfo(torch.uint8)

# %%
# Notebook cell 7
# Information of `8-bit (signed) integer`
torch.iinfo(torch.int8)

# %%
# Notebook cell 8
### Information of `64-bit (signed) integer`
torch.iinfo(torch.int64)

# %%
# Notebook cell 9
### Information of `32-bit (signed) integer`
torch.iinfo(torch.int32)

# %%
# Notebook cell 10
### Information of `16-bit (signed) integer`
torch.iinfo(torch.int16)

# %% [markdown]
# ### Floating Points 

# %%
# Notebook cell 12
# by default, python stores float data in fp64
value = 1/3

# %%
# Notebook cell 13
format(value, '.60f')

# %%
# Notebook cell 14
# 64-bit floating point
tensor_fp64 = torch.tensor(value, dtype = torch.float64)

# %%
# Notebook cell 15
print(f"fp64 tensor: {format(tensor_fp64.item(), '.60f')}")

# %%
# Notebook cell 16
tensor_fp32 = torch.tensor(value, dtype = torch.float32)
tensor_fp16 = torch.tensor(value, dtype = torch.float16)
tensor_bf16 = torch.tensor(value, dtype = torch.bfloat16)

# %%
# Notebook cell 17
print(f"fp64 tensor: {format(tensor_fp64.item(), '.60f')}")
print(f"fp32 tensor: {format(tensor_fp32.item(), '.60f')}")
print(f"fp16 tensor: {format(tensor_fp16.item(), '.60f')}")
print(f"bf16 tensor: {format(tensor_bf16.item(), '.60f')}")

# %%
# Notebook cell 18
# Information of `16-bit brain floating point`
torch.finfo(torch.bfloat16)

# %%
# Notebook cell 19
# Information of `32-bit floating point`
torch.finfo(torch.float32)

# %%
# Notebook cell 20
### Information of `16-bit floating point`
torch.finfo(torch.float16)

# %%
# Notebook cell 21
### Information of `64-bit floating point`
torch.finfo(torch.float64)

# %% [markdown]
# ### Downcasting

# %%
# Notebook cell 23
# random pytorch tensor: float32, size=1000
tensor_fp32 = torch.rand(1000, dtype = torch.float32)

# %% [markdown]
# **Note:** As it is random, the values you get will be different from the video.

# %%
# Notebook cell 25
# first 5 elements of the random tensor
tensor_fp32[:5]

# %%
# Notebook cell 26
# downcast the tensor to bfloat16 using the "to" method
tensor_fp32_to_bf16 = tensor_fp32.to(dtype = torch.bfloat16)

# %%
# Notebook cell 27
tensor_fp32_to_bf16[:5]

# %%
# Notebook cell 28
# tensor_fp32 x tensor_fp32
m_float32 = torch.dot(tensor_fp32, tensor_fp32)

# %%
# Notebook cell 29
m_float32

# %%
# Notebook cell 30
# tensor_fp32_to_bf16 x tensor_fp32_to_bf16
m_bfloat16 = torch.dot(tensor_fp32_to_bf16, tensor_fp32_to_bf16)

# %%
# Notebook cell 31
m_bfloat16

# %% [markdown]
# #### Note
# - You'll use "downcasting" as a simple form of quantization in the next lesson.

# %%
# Notebook cell 33

