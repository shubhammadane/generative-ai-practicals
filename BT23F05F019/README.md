# Generative AI Practicals - BT23F05F019

This folder contains my completed work for all eight experiments in the Generative Artificial Intelligence Laboratory. The notebooks include code, explanations, and saved outputs for evaluation.

## Practical Work

| No. | Experiment | Implementation |
| ---: | --- | --- |
| 1 | Generative AI vs Discriminative Models | Compares a digit classifier with a generative autoencoder using handwritten digit data. |
| 2 | Probabilistic Modeling and Synthetic Data | Fits a Gaussian distribution, samples synthetic data, and compares it with the probability density function. |
| 3 | Transformer-Based Language Models and Text Generation | Demonstrates tokens, embeddings, self-attention, and local next-token text generation. |
| 4 | Prompt Engineering and RAG | Uses TF-IDF retrieval to build a context-grounded prompt for reducing hallucination. |
| 5 | Generative Adversarial Networks | Implements a PyTorch generator and discriminator for synthetic MNIST-style image generation. |
| 6 | Diffusion-Based Image Generation | Uses Stable Diffusion to generate an image from a text prompt and saves the result. |
| 7 | Neural Style Transfer and GAN-Based Art | Creates content and style images and applies feature-based neural style transfer. |
| 8 | Open-Source Generative AI Models | Loads a pretrained language model, prepares a dataset, fine-tunes it, and performs text generation. |

## Files

- [Practical 1](Practical_1_BT23F05F019.ipynb)
- [Practical 2](Practical_2_BT23F05F019.ipynb)
- [Practical 3](Practical_3_BT23F05F019.ipynb)
- [Practical 4](Practical_4_BT23F05F019.ipynb)
- [Practical 5](Practical_5_BT23F05F019.ipynb)
- [Practical 6](Practical_6_BT23F05F019.ipynb)
- [Practical 7](Practical_7_BT23F05F019.ipynb)
- [Practical 8](Practical_8_BT23F05F019.ipynb)
- [requirements.txt](requirements.txt)

## How to Run

From the repository root:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r BT23F05F019\requirements.txt
jupyter notebook
```

Open the required notebook from the `BT23F05F019` folder and run its cells from top to bottom.

## Execution Notes

- Practicals 1-4 use lightweight local demonstrations and can run without a GPU or API key.
- Practicals 5 and 7 use PyTorch and can run on CPU, although training is faster on a CUDA-enabled GPU.
- Practical 6 downloads the Stable Diffusion model from Hugging Face and requires internet access, substantial disk space, and preferably a GPU.
- Practical 8 downloads `distilgpt2` and uses the Hugging Face `datasets` and `transformers` libraries. Internet access is required the first time the model is loaded.
- Random seeds are set where applicable for reproducible results.
- Previously generated outputs are stored in the notebooks so the work can be reviewed without immediately rerunning every cell.

## Submission Checklist

- Eight practical notebooks are included.
- Each practical contains executable Python code and output.
- This README documents the aim and implementation of every practical.
- `requirements.txt` lists the Python dependencies.
- No virtual environment, downloaded model cache, or temporary dataset folder is included in the submission.
