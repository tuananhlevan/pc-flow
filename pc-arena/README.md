# Probabilistic Circuit Arena
Probabilistic Circuit Arena is a flexible training framework for [Probabilistic Circuits](https://starai.cs.ucla.edu/papers/ProbCirc20.pdf) built on top of [PyJuice](https://github.com/Tractables/pyjuice?tab=readme-ov-file). It provides a unified and easy-to-use interface for experimenting with various learning algorithms (e.g. Full-batch EM, SGD, Mini-batch EM, Anemone) and circuit structures (e.g. HMMs, HCLTs).

## Installation

The framework is built on [PyJuice](https://github.com/Tractables/pyjuice?tab=readme-ov-file), PyTorch Lightning etc. To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Example Usage

You can start training a probabilistic circuit with a single command.  
For example:

```bash
python main_ddp.py --model-config hmm_128_256 --data-config wikitext --optim-config anemone
```

This command launches distributed training using the specified model, dataset, and optimizer configurations.

All configuration files are stored under the [`configs/`](https://github.com/liuanji/pc-arena/tree/main/configs) directory:

- `configs/model/` — model structure (e.g., HMM, HCLT)  
- `configs/data/` — dataset settings and preprocessing options  
- `configs/optim/` — optimizer type and learning hyperparameters  

You can customize experiments by modifying the corresponding `.yaml` files or creating new ones to define your own setups.

### Preparing Datasets

Some datasets require preprocessing. To do so, run the corresponding file under `src/data/`.

## Example Results

Below are example results obtained by running **PC-Arena** using different optimizers and probabilistic circuit architectures across several datasets.  
The best result in each column is **bolded**.

---

### Table 1: NLLs on the UK BioBank Chromosome 6 dataset

| **Optimizer**     | **HCLT 512** | **HCLT 1024** | **PDHCLT 512** | **PDHCLT 1024** |
|--------------------|--------------|---------------|----------------|-----------------|
| Full-batch EM      | 55.3         | 53.8          | 46.5           | 45.1            |
| Adam               | 102.7        | 100.4         | 112.4          | 110.4           |
| Mini-batch EM      | 55.7         | 55.5          | 49.5           | 47.2            |
| Anemone            | **54.5**     | **52.1**      | **45.3**       | **42.2**        |

---

### Table 2: NLLs on the ImageNet32 dataset’s validation subset

| **Optimizer**     | **PDHCLT 256** | **HCLT 512** | **HCLT 1024** | **HCLT 512 (YCC)** | **HCLT 1024 (YCC)** |
|--------------------|----------------|---------------|---------------|--------------------|---------------------|
| Full-batch EM      | **2529**       | 2480          | **2469**      | 2164               | 2163                |
| Adam               | 2553           | 2518          | OOM           | 2187               | OOM                 |
| Mini-batch EM      | **2529**       | 2506          | 2470          | 2179               | 2232                |
| Anemone            | 2530           | **2477**      | **2469**      | **2158**           | **2159**            |

---

### Table 3: NLLs on the WikiText-103 dataset

| **Optimizer**     | **HMM 256** | **HMM 512** | **HMM 1024** | **Monarch HMM 1024** |
|--------------------|-------------|--------------|--------------|----------------------|
| Full-batch EM      | 722.6       | 702.2        | 682.8        | 738.1                |
| Adam               | 735.7       | OOM          | OOM          | OOM                  |
| Mini-batch EM      | 725.2       | 703.2        | **682.1**    | 734.6                |
| Anemone            | **722.2**   | **701.7**    | 682.3        | **734.0**            |
