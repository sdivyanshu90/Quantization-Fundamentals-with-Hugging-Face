# %% [markdown]
# Source notebook: Models_with_different_Data_Types.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

# %% [markdown]
# # Lesson 3: Loading ML Models with Different Data Types
# 
# In this lab, you will load ML models in different datatypes.

# %% [markdown]
# - Load the Dummy Model from the helper file.
# - To access the `helper.py` file, you can click `File --> Open...`, on the top left.

# %%
# Notebook cell 3
#123456789#123456789#123456789#123456789#123456789#123456789#123456789

# %%
# Notebook cell 4
from helper import DummyModel

# %%
# Notebook cell 5
model = DummyModel()

# %%
# Notebook cell 6
model

# %% [markdown]
# - Create a function to inspect the data types of the parameters in a model.

# %%
# Notebook cell 8
def print_param_dtype(model):
    for name, param in model.named_parameters():
        print(f"{name} is loaded in {param.dtype}")

# %%
# Notebook cell 9
print_param_dtype(model)

# %% [markdown]
# ## Model Casting: `float16`
# 
# - Cast the model into a different precision.

# %%
# Notebook cell 11
# float 16
model_fp16 = DummyModel().half()

# %% [markdown]
# - Inspect the data types of the parameters.

# %%
# Notebook cell 13
print_param_dtype(model_fp16)

# %%
# Notebook cell 14
model_fp16

# %% [markdown]
# - Run simple inference using model.

# %%
# Notebook cell 16
import torch

# %%
# Notebook cell 17
dummy_input = torch.LongTensor([[1, 0], [0, 1]])

# %%
# Notebook cell 18
# inference using float32 model
logits_fp32 = model(dummy_input)

# %%
# Notebook cell 19
logits_fp32

# %%
# Notebook cell 20
# inference using float16 model
try:
    logits_fp16 = model_fp16(dummy_input)
except Exception as error:
    print("\033[91m", type(error).__name__, ": ", error, "\033[0m")

# %% [markdown]
# ## Model Casting: `bfloat16`
# 
# #### Note about deepcopy
# - `copy.deepcopy` makes a copy of the model that is independent of the original.  Modifications you make to the copy will not affect the original, because you're making a "deep copy".  For more details, see the Python docs on the [copy][https://docs.python.org/3/library/copy.html] library.

# %%
# Notebook cell 22
from copy import deepcopy

# %%
# Notebook cell 23
model_bf16 = deepcopy(model)

# %%
# Notebook cell 24
model_bf16 = model_bf16.to(torch.bfloat16)

# %%
# Notebook cell 25
print_param_dtype(model_bf16)

# %%
# Notebook cell 26
logits_bf16 = model_bf16(dummy_input)

# %% [markdown]
# - Now, compare the difference between `logits_fp32` and `logits_bf16`.

# %%
# Notebook cell 28
mean_diff = torch.abs(logits_bf16 - logits_fp32).mean().item()
max_diff = torch.abs(logits_bf16 - logits_fp32).max().item()

print(f"Mean diff: {mean_diff} | Max diff: {max_diff}")

# %% [markdown]
# ## Using Popular Generative Models in Different Data Types

# %% [markdown]
# - Load [Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base) to perform image captioning.
# 
# #### To get the sample code that Younes showed:
# - Click on the "Model Card" tab.
# - On the right, click on the button "<> Use in Transformers", you'll see a popup with sample code for loading this model.
# 
# ```Python
# # Load model directly
# from transformers import AutoProcessor, AutoModelForSeq2SeqLM
# 
# processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/blip-image-captioning-base")
# ```
# 
# - To see the sample code with an example, click on "Read model documentation" at the bottom of the popup.  It opens a new tab.
#   https://huggingface.co/docs/transformers/main/en/model_doc/blip#transformers.BlipForConditionalGeneration
# - On this page, scroll down a bit, past the "parameters", section, and you'll see "Examples:"
# 
# ```Python
# from PIL import Image
# import requests
# from transformers import AutoProcessor, BlipForConditionalGeneration
# 
# processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
# 
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# text = "A picture of"
# 
# inputs = processor(images=image, text=text, return_tensors="pt")
# 
# outputs = model(**inputs)
# ```

# %%
# Notebook cell 31
from transformers import BlipForConditionalGeneration

# %%
# Notebook cell 32
model_name = "Salesforce/blip-image-captioning-base"

# %%
# Notebook cell 33
model = BlipForConditionalGeneration.from_pretrained(model_name)

# %%
# Notebook cell 34
# inspect the default data types of the model

# print_param_dtype(model)

# %% [markdown]
# - Check the memory footprint of the model. 

# %%
# Notebook cell 36
fp32_mem_footprint = model.get_memory_footprint()

# %%
# Notebook cell 37
print("Footprint of the fp32 model in bytes: ",
      fp32_mem_footprint)
print("Footprint of the fp32 model in MBs: ", 
      fp32_mem_footprint/1e+6)

# %% [markdown]
# - Load the same model in `bfloat16`.

# %%
# Notebook cell 39
model_bf16 = BlipForConditionalGeneration.from_pretrained(
                                               model_name,
                               torch_dtype=torch.bfloat16
)

# %%
# Notebook cell 40
bf16_mem_footprint = model_bf16.get_memory_footprint()

# %%
# Notebook cell 41
# Get the relative difference
relative_diff = bf16_mem_footprint / fp32_mem_footprint

print("Footprint of the bf16 model in MBs: ", 
      bf16_mem_footprint/1e+6)
print(f"Relative diff: {relative_diff}")

# %% [markdown]
# ### Model Performance: `float32` vs `bfloat16`

# %% [markdown]
# - Now, compare the generation results of the two model.

# %%
# Notebook cell 44
from transformers import BlipProcessor

# %%
# Notebook cell 45
processor = BlipProcessor.from_pretrained(model_name)

# %% [markdown]
# - Load the image.

# %%
# Notebook cell 47
from helper import load_image, get_generation
from IPython.display import display

img_url = 'https://storage.googleapis.com/\
sfr-vision-language-research/BLIP/demo.jpg'

image = load_image(img_url)
display(image.resize((500, 350)))

# %%
# Notebook cell 48
results_fp32 = get_generation(model, 
                              processor, 
                              image, 
                              torch.float32)

# %%
# Notebook cell 49
print("fp32 Model Results:\n", results_fp32)

# %%
# Notebook cell 50
results_bf16 = get_generation(model_bf16, 
                              processor, 
                              image, 
                              torch.bfloat16)

# %%
# Notebook cell 51
print("bf16 Model Results:\n", results_bf16)

# %% [markdown]
# ### Default Data Type
# 
# - For Hugging Face Transformers library, the deafult data type to load the models in is `float32`
# - You can set the "default data type" as what you want.

# %%
# Notebook cell 53
desired_dtype = torch.bfloat16
torch.set_default_dtype(desired_dtype)

# %%
# Notebook cell 54
dummy_model_bf16 = DummyModel()

# %%
# Notebook cell 55
print_param_dtype(dummy_model_bf16)

# %% [markdown]
# - Similarly, you can reset the default data type to float32.

# %%
# Notebook cell 57
torch.set_default_dtype(torch.float32)

# %%
# Notebook cell 58
print_param_dtype(dummy_model_bf16)

# %% [markdown]
# ### Note
# - You just used a simple form of quantization, in which the model's parameters are saved in a more compact data type (bfloat16).  During inference, the model performs its calculations in this data type, and its activations are in this data type.
# - In the next lesson, you will use another quantization method, "linear quantization", which enables the quantized model to maintain performance much closer to the original model by converting from the compressed data type back to the original FP32 data type during inference.

# %%
# Notebook cell 60

