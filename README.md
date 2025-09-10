# Quantization Fundamentals with Hugging Face

Welcome to the repository for the **[Quantization Fundamentals with Hugging Face](https://learn.deeplearning.ai/courses/quantization-fundamentals/)** course, brought to you by **[DeepLearning.AI](https://www.deeplearning.ai/)** in collaboration with **[Hugging Face](https://huggingface.co/)**. 🎉

This repo contains the **notebooks and explanations** used in the course to help you understand how to make **Large Language Models (LLMs)** faster, lighter, and more efficient through **quantization**.

---

## 📖 About the Course

Large deep learning models are powerful but often **too big to run efficiently**. Quantization offers a practical way to reduce memory, storage, and compute requirements while keeping model performance intact.

By the end of this course, you will:

* ✅ Understand how to **work with big models**
* ✅ Learn about **data types and number representations** in deep learning
* ✅ Explore how to **load models efficiently with different data types**
* ✅ Grasp the **theory of quantization** and why it works
* ✅ Apply **quantization to LLMs** for real-world usage

---

## 📚 Course Topics

### 1. Handling Big Models

Large models create **memory and compute challenges**.

* **Weights, activations, optimizer state** all contribute to memory usage.
* Techniques: gradient checkpointing, offloading to CPU/NVMe, model parallelism, ZeRO optimization.
* Precision reduction (FP16, INT8) significantly lowers memory.
* Example: A 7B parameter model takes \~28GB in FP32, \~14GB in FP16, \~7GB in INT8.

---

### 2. Data Types and Sizes

Quantization depends on how numbers are represented.

* **FP32**: 32-bit float, high precision, high memory.
* **FP16 / BF16**: 16-bit float formats, half the storage, BF16 has better stability.
* **INT8 / INT4**: integer formats, require mapping floats to integers using `scale` and `zero_point`.
* **Per-tensor vs per-channel** quantization: per-channel usually reduces errors better for weights.

---

### 3. Loading Models by Data Type

Efficient model loading saves memory and computation.

* Load directly in FP16/BF16 using `torch_dtype` in Hugging Face.
* Load in INT8/INT4 using quantization libraries (`load_in_8bit=True`).
* Use **mixed precision** (keep sensitive layers like LayerNorm in FP16, others quantized).
* `device_map` can spread model layers across multiple GPUs or CPUs for large models.

---

### 4. Quantization Theory

Quantization approximates floating-point numbers with integers.

* **Uniform affine quantization**:
  `x ≈ scale * (q - zero_point)`
* **Symmetric vs asymmetric**: symmetric (zero\_point=0) works for weights; asymmetric better for activations.
* **PTQ (Post-Training Quantization)**: quantize after training, fast but may lose accuracy.
* **QAT (Quantization-Aware Training)**: simulate quantization during training to preserve accuracy.
* **Calibration**: Use a representative dataset to set activation ranges and avoid poor scaling.

---

### 5. Quantization of LLMs

Applying quantization to **transformer-based LLMs** requires care.

* **Quantize weights (INT8/INT4)** → major memory savings.
* **Keep sensitive layers in FP16/BF16** (LayerNorm, softmax, embeddings).
* **Steps**: baseline evaluation → choose precision → PTQ with calibration → evaluate → refine (per-channel, mixed precision) → deploy.
* **Pitfalls**: bad calibration, quantizing sensitive ops, skipping evaluation.
* **Results**: INT8 weights cut memory 4×, INT4 cuts 8×, with little accuracy loss if done properly.

---

## 📚 Recommended Resources

* 🌐 [Hugging Face Documentation](https://huggingface.co/docs)
* 📘 [DeepLearning.AI Courses](https://www.deeplearning.ai/courses/)
* 📄 [Bits and Bytes Library (Hugging Face)](https://huggingface.co/docs/bitsandbytes/index)
* 🎥 [YouTube: Hugging Face Tutorials](https://www.youtube.com/c/huggingface)

---

## ⭐ Acknowledgements

This course is created by **[DeepLearning.AI](https://www.deeplearning.ai/)** and **[Hugging Face](https://huggingface.co/)**, with contributions from the open-source community. 🚀

---

✨ *Star this repo if you find it useful and keep exploring quantization!* ✨
