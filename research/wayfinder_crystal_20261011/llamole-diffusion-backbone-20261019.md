# Llamole diffusion and language backbones

Primary-source check of arXiv:2410.04223v1 and its linked [official Llamole repository](https://github.com/liugangcode/Llamole), 2026-10-19.

Llamole is an autoregressive language model orchestrating separate graph models. The LLM writes text and uses trigger/query tokens to call GraphDiT for molecule graphs or a GNN for reaction-template prediction; GraphDiT performs discrete diffusion over graph atoms and bonds. It is not an LLM diffusion backbone. See paper §3 and Appendix C.1.

The paper reports GraphDiT at 28 Transformer layers, hidden size 1,024, 16 heads, MLP size 4,096, and about 574M parameters. It conditions through AdaLN on text and explicit molecular properties. The authors pre-train it for 45 epochs on roughly 600K graph–text pairs paired with eight property labels: about 300K small-molecule and 300K polymer pairs in MolPair, excluding MolQA; the reported run took about a week on one A100 (Appendix B.3 and C.1). The official repo and [model card](https://huggingface.co/liuganghuggingface/Llamole-Pretrained-GraphDiT) expose GraphDiT as a pretrained graph decoder/checkpoint. During Llamole multimodal SFT, the paper freezes GraphDiT and the other graph modules while tuning the language model and connectors; it reports LoRA tuning for four epochs (Appendix C.3).

The language backbones are existing pretrained checkpoints, not models trained from scratch for Llamole: Llama-3.1-8B-Instruct, Qwen2-7B-Instruct, and Mistral-7B-Instruct-v0.3. The paper reports these three LoRA variants; the repo README names the same base models and says pretrained graph modules are downloaded for SFT (Appendix C.3; repo README, “Automatic Model Download” and “Supervised Fine-Tuning”).

## Distinction from PLaID and the local experiment

PLaID/PLaID++ uses Qwen2.5-7B as an autoregressive crystal-text model; it has no diffusion backbone. Both [v1 §4.1](https://arxiv.org/html/2509.07150v1) and [v4 §4.1](https://arxiv.org/html/2509.07150v4) identify this base. The [official training implementation](https://github.com/andaero/PLaID/blob/main/llm_finetune.py) loads `AutoModelForCausalLM`; its infilling examples remain causal answer generation rather than a diffusion process.

Our shared-model run initializes from Qwen3-8B, not Llama 3, PLaID weights, or a pretrained graph/crystal diffusion checkpoint. It adapts the same core to causal text and masked crystal denoising, using PLaID-style Wyckoff text. The completed pure-AR adapter was not its initializer. See the [local frozen protocol](../../experiments/structure_conditioned_hybrid8b_20261018/PROTOCOL.md). This is a transfer design requiring validation, not a direct reproduction of either paper's structure generator.

**Boundary of evidence:** the paper documents a GraphDiT pretraining run and the repo distributes/loads its pretrained checkpoint. These sources do not establish whether that run initialized GraphDiT randomly or continued from an earlier Graph-DiT checkpoint; “trained from scratch” at initialization is therefore unconfirmed.

Sources: [arXiv paper, v1](https://arxiv.org/html/2410.04223v1) (§3, Appendix B.3, C.1, C.3); [official Llamole repository](https://github.com/liugangcode/Llamole) (README, “Automatic Model Download” / “Supervised Fine-Tuning”); [official GraphDiT checkpoint](https://huggingface.co/liuganghuggingface/Llamole-Pretrained-GraphDiT).
