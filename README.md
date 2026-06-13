# Brain-Inspired Generative Replay for Class-Incremental Learning

A PyTorch implementation of a generative-replay-based continual learning
system that mitigates catastrophic forgetting on the Split MNIST benchmark.
The approach combines a class-conditional Variational Autoencoder (VAE)
with knowledge distillation to "rehearse" previously learned classes
without storing any real data from past tasks.

## Overview

In class-incremental learning, a model is trained on a sequence of tasks,
each introducing new classes, without access to data from earlier tasks.
Naively fine-tuning a neural network in this setting causes **catastrophic
forgetting** — performance on earlier tasks collapses as the model
specializes on the current one.

This project addresses that problem using **generative replay**: instead
of storing real samples from old tasks (which may be infeasible due to
memory or privacy constraints), the model learns to generate synthetic
samples of old classes from a learned latent prior, and uses these
pseudo-samples to "remind" itself of past tasks while learning a new one.

## Approach

- **Model**: A conditional VAE with an auxiliary classifier head, sharing
  encoder features between reconstruction and classification.
- **Class-conditional latent priors**: Each class has its own learnable
  Gaussian prior in latent space, used both for KL regularization during
  training and for sampling pseudo-data during replay.
- **Generative replay**: Before training on a new task, a frozen snapshot
  ("teacher") of the current model is saved. During training, the teacher
  generates pseudo-samples of previously seen classes, which are mixed
  into training alongside real data from the current task.
- **Knowledge distillation**: The teacher's soft predictions on replayed
  samples are distilled into the student model via a temperature-scaled
  KL loss, in addition to the VAE objective on the replayed data.
- **Loss balancing**: The relative weight of current-task vs. replay loss
  is set in proportion to the number of new vs. previously seen classes,
  ensuring older tasks aren't drowned out as more tasks accumulate.

## Experimental Setup

- **Dataset**: Split MNIST — the 10 MNIST digit classes divided into 5
  sequential tasks of 2 classes each.
- **Architecture**: VAE with two 400-unit hidden layers and a 100-dimensional
  latent space.
- **Training**: 20 epochs per task, batch size 128, Adam optimizer (lr = 0.001),
  replay batch size = 3x the current batch size.

## Results

After sequentially training on all 5 tasks:

| Metric | Value |
|---|---|
| Final average accuracy across all tasks | **96.66%** |
| Average catastrophic forgetting | **2.17%** |
| Reference (paper, Fig. 3c) | ~95% |

**Per-task accuracy after training on Task 5:**

| Task | Accuracy |
|---|---|
| Task 1 | 99.53% |
| Task 2 | 92.41% |
| Task 3 | 95.57% |
| Task 4 | 97.53% |
| Task 5 | 98.23% |

**Task-wise accuracy matrix** (accuracy on each task, evaluated after
training on each subsequent task):

| After Task ↓ / Eval Task → | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| 1 | 100 | - | - | - | - |
| 2 | 100 | 98 | - | - | - |
| 3 | 100 | 97 | 99 | - | - |
| 4 | 100 | 96 | 98 | 99 | - |
| 5 | 100 | 92 | 96 | 98 | 98 |

The diagonal-to-final-column comparison shows minimal degradation in
earlier tasks even after learning four additional tasks, demonstrating
that generative replay effectively preserves prior knowledge.

A full results figure (`results/final_results.png`) shows the accuracy
matrix, the learning curve across tasks, and per-task forgetting —
see [Adding Results](#adding-results-from-colab) below for how to add this.

## Project Structure

```
.
├── generative_replay.py   # Main training and evaluation script
├── results/
│   └── final_results.png  # Output plots (accuracy matrix, learning curve, forgetting)
└── README.md
```

## Setup

```bash
pip install torch torchvision numpy matplotlib tqdm
```

## Usage

```bash
python generative_replay.py
```

This will download MNIST automatically, train sequentially across 5 tasks,
print per-task evaluation metrics after each stage, and save a results
figure to `final_results.png`.

To change the experiment configuration, edit the call at the bottom of
the script:

```python
learner, task_accs, avg_accs = run_experiment(
    num_tasks=5,
    epochs=20,
    batch_size=128,
    beta=1.0
)
```

## Adding Results from Colab

Since the run was done on Google Colab and the local results folder is no
longer available, here's how to get the output files into your repo:

1. **If the Colab notebook/session is still accessible**: re-open it, and
   in a new cell run:
   ```python
   from google.colab import files
   files.download('final_results.png')
   ```
   This downloads the file to your computer. Then drag it into the
   `results/` folder in your local repo (or upload directly via the GitHub
   web UI: open the repo → Add file → Upload files → drop `final_results.png`
   into a `results/` folder).

2. **If the Colab session has expired but the notebook (.ipynb) is saved**:
   re-run the notebook (Runtime → Run all) — since the script saves
   `final_results.png` automatically via `plt.savefig(...)`, re-running
   will regenerate it, which you can then download as above.

3. **If you only have the printed text output** (like the accuracy numbers
   you shared), you can still document results in the README as a table
   (already done above) even without the image — the table communicates
   the key numbers clearly to anyone reviewing the repo.

4. **Recommended going forward**: at the end of your Colab notebook, add
   a cell that saves all outputs to Google Drive (`from google.colab import
   drive; drive.mount('/content/drive')` then save paths under
   `/content/drive/MyDrive/...`), so results persist even if the Colab
   runtime disconnects.

## Future Work

- Extend to more challenging benchmarks (Split CIFAR-10/100)
- Compare against other continual learning baselines (EWC, iCaRL)
- Explore conditional generative replay with diffusion-based generators
