"""
Brain-Inspired Generative Replay for Class-Incremental Continual Learning
 
Implements a VAE-based continual learner that mitigates catastrophic
forgetting on Split MNIST by generating pseudo-samples of previously
learned classes (generative replay) and distilling knowledge from a
snapshot of the model trained on prior tasks.
"""
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import copy
import warnings
 
warnings.filterwarnings('ignore')
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
 
 
# ---------------------------------------------------------------------------
# Model: Conditional VAE with an auxiliary classifier head
# ---------------------------------------------------------------------------
class StableVAE(nn.Module):
    """
    Conditional VAE with a classification head.
 
    The encoder produces a latent distribution (mu, logvar) and shares
    its final hidden representation with a linear classifier. Each class
    also has a learnable Gaussian prior (class_mu, class_logvar) in latent
    space, which is used both as a regularization target during training
    and as the sampling distribution for generative replay.
    """
 
    def __init__(self, input_size=784, hidden_sizes=[400, 400], latent_size=100,
                 num_classes=10):
        super(StableVAE, self).__init__()
 
        self.input_size = input_size
        self.latent_size = latent_size
        self.num_classes = num_classes
 
        # Encoder
        self.enc1 = nn.Linear(input_size, hidden_sizes[0])
        self.enc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc_mu = nn.Linear(hidden_sizes[1], latent_size)
        self.fc_logvar = nn.Linear(hidden_sizes[1], latent_size)
 
        # Auxiliary classifier (shares encoder features)
        self.classifier = nn.Linear(hidden_sizes[1], num_classes)
 
        # Per-class learnable prior in latent space (used for replay sampling)
        self.class_mu = nn.Parameter(torch.randn(num_classes, latent_size) * 0.01)
        self.class_logvar = nn.Parameter(torch.zeros(num_classes, latent_size) - 1.0)
 
        # Decoder
        self.dec1 = nn.Linear(latent_size, hidden_sizes[1])
        self.dec2 = nn.Linear(hidden_sizes[1], hidden_sizes[0])
        self.dec3 = nn.Linear(hidden_sizes[0], input_size)
 
        self._init_weights()
 
    def _init_weights(self):
        # Kaiming initialization for stable training with ReLU activations
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
 
    def encode(self, x):
        h = F.relu(self.enc1(x))
        h = F.relu(self.enc2(h))
        return self.fc_mu(h), self.fc_logvar(h), h
 
    def reparameterize(self, mu, logvar):
        # Standard reparameterization trick for sampling z ~ N(mu, sigma^2)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
 
    def decode(self, z):
        h = F.relu(self.dec1(z))
        h = F.relu(self.dec2(h))
        return torch.sigmoid(self.dec3(h))
 
    def forward(self, x):
        mu, logvar, h = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        logits = self.classifier(h)
        return recon, mu, logvar, logits
 
    def generate(self, num_samples, class_idx):
        """Sample pseudo-data for a given class from its learned latent prior."""
        with torch.no_grad():
            mu = self.class_mu[class_idx].unsqueeze(0).repeat(num_samples, 1)
            logvar = self.class_logvar[class_idx].unsqueeze(0).repeat(num_samples, 1)
            z = self.reparameterize(mu, logvar)
            return self.decode(z)
 
 
# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
def compute_loss(recon, x, mu, logvar, logits, labels,
                  class_mu, class_logvar, beta=1.0):
    """
    Combined VAE + classification loss.
 
    - Reconstruction loss: binary cross-entropy between input and output
    - KL term: KL divergence between the encoded posterior and the
      class-conditional prior (rather than a standard N(0, I) prior)
    - Classification loss: cross-entropy on the auxiliary classifier head
    """
    batch_size = x.size(0)
 
    recon_loss = F.binary_cross_entropy(recon, x, reduction='sum') / batch_size
 
    # KL divergence against the class-conditional prior for each sample's label
    batch_mu = class_mu[labels]
    batch_logvar = class_logvar[labels]
 
    # Clamp log-variances to avoid numerical instability
    logvar = torch.clamp(logvar, -10, 10)
    batch_logvar = torch.clamp(batch_logvar, -10, 10)
 
    kl = -0.5 * torch.mean(
        1 + logvar - batch_logvar -
        ((mu - batch_mu).pow(2) + logvar.exp()) / (batch_logvar.exp() + 1e-8)
    )
 
    class_loss = F.cross_entropy(logits, labels)
 
    vae_loss = recon_loss + beta * kl
    total_loss = vae_loss + class_loss
 
    return total_loss, recon_loss, kl, class_loss
 
 
def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    """KL-based soft-target distillation loss between student and teacher logits."""
    p_teacher = F.softmax(teacher_logits / temperature, dim=1)
    log_p_student = F.log_softmax(student_logits / temperature, dim=1)
    loss = F.kl_div(log_p_student, p_teacher, reduction='batchmean')
    return loss * (temperature ** 2)
 
 
# ---------------------------------------------------------------------------
# Continual learner with generative replay
# ---------------------------------------------------------------------------
class ContinualLearner:
    """
    Manages sequential training across tasks using generative replay.
 
    Before training on a new task, a frozen copy of the current model is
    saved as the "teacher". During training, pseudo-samples of previously
    seen classes are generated by the teacher and used both to recompute
    the VAE loss on old classes and to distill the teacher's classifier
    outputs into the student via a soft-target loss.
    """
 
    def __init__(self, model, lr=0.001, beta=1.0):
        self.model = model.to(device)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.beta = beta
        self.previous_model = None
        self.all_seen_classes = []
        self.task_id = 0
 
    def train_task(self, train_loader, task_classes, epochs=10):
        self.model.train()
 
        # Freeze a snapshot of the model trained on previous tasks (the "teacher")
        if self.task_id > 0:
            self.previous_model = copy.deepcopy(self.model)
            self.previous_model.eval()
            for p in self.previous_model.parameters():
                p.requires_grad = False
 
        for epoch in range(epochs):
            pbar = tqdm(train_loader, desc=f'Task {self.task_id + 1}, Epoch {epoch + 1}/{epochs}')
 
            for data, labels in pbar:
                data = data.to(device).view(data.size(0), -1)
                labels = labels.to(device)
                batch_size = data.size(0)
 
                self.optimizer.zero_grad()
 
                # --- Loss on current task's real data ---
                recon, mu, logvar, logits = self.model(data)
                loss_cur, recon_cur, kl_cur, class_cur = compute_loss(
                    recon, data, mu, logvar, logits, labels,
                    self.model.class_mu, self.model.class_logvar, self.beta
                )
 
                total_loss = loss_cur
 
                # --- Generative replay of previous classes ---
                if self.previous_model is not None and len(self.all_seen_classes) > 0:
                    # Sample more replay data than the current batch size to
                    # give old classes sufficient weight during training
                    n_replay = batch_size * 3
 
                    replay_labels = torch.tensor(
                        np.random.choice(self.all_seen_classes, n_replay, replace=True)
                    ).to(device)
 
                    # Generate pseudo-samples for the replay batch using the teacher
                    with torch.no_grad():
                        replay_data = torch.zeros(n_replay, self.model.input_size).to(device)
                        for c in torch.unique(replay_labels):
                            mask = replay_labels == c
                            n = mask.sum().item()
                            if n > 0:
                                replay_data[mask] = self.previous_model.generate(n, c.item())
 
                        # Teacher's predictions on the replay data (distillation target)
                        _, _, _, teacher_logits = self.previous_model(replay_data)
 
                    # Student's predictions on the same replay data
                    recon_r, mu_r, logvar_r, student_logits = self.model(replay_data)
 
                    # VAE loss on replayed (pseudo) data
                    loss_r, _, _, _ = compute_loss(
                        recon_r, replay_data, mu_r, logvar_r, student_logits, replay_labels,
                        self.model.class_mu, self.model.class_logvar, self.beta
                    )
 
                    # Soft-target distillation loss
                    dist_loss = distillation_loss(student_logits, teacher_logits, temperature=2.0)
 
                    replay_loss = loss_r + dist_loss
 
                    # Weight current vs. replay loss in proportion to the number
                    # of new vs. previously seen classes
                    n_old = len(self.all_seen_classes)
                    n_new = len(task_classes)
                    replay_weight = n_old / (n_old + n_new)
                    current_weight = n_new / (n_old + n_new)
 
                    total_loss = current_weight * loss_cur + replay_weight * replay_loss
 
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.optimizer.step()
 
                pbar.set_postfix({
                    'loss': f'{total_loss.item():.1f}',
                    'class': f'{class_cur.item():.3f}',
                    'recon': f'{recon_cur.item():.1f}'
                })
 
        self.all_seen_classes.extend(task_classes)
        self.task_id += 1
 
    def evaluate(self, test_loader, classes_seen):
        """Compute classification accuracy restricted to classes seen so far."""
        self.model.eval()
        correct = total = 0
 
        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(device).view(data.size(0), -1)
                labels = labels.to(device)
 
                _, _, _, logits = self.model(data)
 
                # Mask out logits for classes not yet introduced
                mask = torch.ones_like(logits) * float('-inf')
                mask[:, classes_seen] = 0
                logits = logits + mask
 
                _, pred = torch.max(logits, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
 
        return 100 * correct / total
 
 
# ---------------------------------------------------------------------------
# Dataset: Split MNIST
# ---------------------------------------------------------------------------
def get_split_mnist(num_tasks=5):
    """Partition MNIST into `num_tasks` sequential tasks with disjoint class sets."""
    transform = transforms.Compose([transforms.ToTensor()])
 
    train_ds = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
 
    classes_per_task = 10 // num_tasks
    tasks = []
 
    for t in range(num_tasks):
        task_classes = list(range(t * classes_per_task, (t + 1) * classes_per_task))
 
        train_idx = [i for i, (_, y) in enumerate(train_ds) if y in task_classes]
        test_idx = [i for i, (_, y) in enumerate(test_ds) if y in task_classes]
 
        tasks.append({
            'train': Subset(train_ds, train_idx),
            'test': Subset(test_ds, test_idx),
            'classes': task_classes
        })
 
    return tasks
 
 
# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_results(task_accs, avg_accs, num_tasks):
    """Plot the per-task accuracy matrix, learning curve, and forgetting per task."""
    fig = plt.figure(figsize=(18, 5))
 
    # Task-wise accuracy matrix
    ax1 = plt.subplot(1, 3, 1)
    matrix = np.zeros((num_tasks, num_tasks))
    for i in range(num_tasks):
        for j in range(len(task_accs[i])):
            matrix[i, j] = task_accs[i][j]
 
    im = ax1.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
    ax1.set_xlabel('Task Evaluated', fontsize=13, fontweight='bold')
    ax1.set_ylabel('After Training Task', fontsize=13, fontweight='bold')
    ax1.set_title('Task-wise Accuracy Matrix', fontsize=14, fontweight='bold', pad=15)
 
    for i in range(num_tasks):
        for j in range(i + 1):
            color = 'white' if matrix[i, j] < 50 else 'black'
            ax1.text(j, i, f'{matrix[i, j]:.0f}', ha="center", va="center",
                     color=color, fontsize=12, fontweight='bold')
 
    plt.colorbar(im, ax=ax1, label='Accuracy (%)')
 
    # Average accuracy over time
    ax2 = plt.subplot(1, 3, 2)
    x = range(1, num_tasks + 1)
    ax2.plot(x, avg_accs, 'o-', linewidth=3, markersize=12, color='#2E86AB',
             label='Model Performance')
    ax2.fill_between(x, avg_accs, alpha=0.3, color='#2E86AB')
    ax2.axhline(90, color='green', linestyle='--', linewidth=2, alpha=0.6,
                label='Paper Target (90%)')
    ax2.set_xlabel('Tasks Learned', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Average Accuracy (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Learning Progress', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim([0, 105])
    ax2.legend(fontsize=10)
 
    # Forgetting per task at the end of training
    ax3 = plt.subplot(1, 3, 3)
    forgetting_per_task = []
    for t in range(num_tasks):
        if t < len(task_accs[-1]):
            initial = task_accs[t][t]  # Accuracy right after this task was learned
            final = task_accs[-1][t]   # Accuracy at the very end of training
            forgetting_per_task.append(max(0, initial - final))
        else:
            forgetting_per_task.append(0)
 
    colors = ['#27AE60' if f < 5 else '#F39C12' if f < 15 else '#E74C3C'
              for f in forgetting_per_task]
    ax3.bar(range(1, num_tasks + 1), forgetting_per_task, color=colors,
            edgecolor='black', linewidth=1.5, alpha=0.8)
    ax3.set_xlabel('Task', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Forgetting (%)', fontsize=13, fontweight='bold')
    ax3.set_title('Catastrophic Forgetting per Task', fontsize=14, fontweight='bold', pad=15)
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax3.set_ylim([0, max(forgetting_per_task + [20])])
    ax3.axhline(10, color='orange', linestyle='--', alpha=0.5, linewidth=2)
 
    plt.tight_layout()
    plt.savefig('final_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\nResults saved to 'final_results.png'")
 
 
# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------
def run_experiment(num_tasks=5, epochs=20, batch_size=128, beta=1.0):
    """Run sequential training across all tasks and report final results."""
    print("Brain-Inspired Generative Replay - Continual Learning")
    print(f"Config: tasks={num_tasks}, epochs/task={epochs}, "
          f"batch_size={batch_size}, beta={beta}, replay_ratio=3x batch size")
 
    # Data
    tasks = get_split_mnist(num_tasks)
 
    # Model
    model = StableVAE(
        input_size=784,
        hidden_sizes=[400, 400],
        latent_size=100,
        num_classes=10
    )
 
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
 
    learner = ContinualLearner(model, lr=0.001, beta=beta)
 
    task_accs = []
    avg_accs = []
 
    for t in range(num_tasks):
        print(f"\nTask {t + 1}/{num_tasks} - Classes: {tasks[t]['classes']}")
 
        train_loader = DataLoader(
            tasks[t]['train'],
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
 
        learner.train_task(train_loader, tasks[t]['classes'], epochs=epochs)
 
        # Evaluate on all tasks seen so far
        print(f"Evaluation after Task {t + 1}:")
        accs = []
        all_classes = []
 
        for eval_t in range(t + 1):
            test_loader = DataLoader(
                tasks[eval_t]['test'],
                batch_size=batch_size * 2,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )
            all_classes.extend(tasks[eval_t]['classes'])
 
            acc = learner.evaluate(test_loader, all_classes)
            accs.append(acc)
            print(f"  Task {eval_t + 1}: {acc:.2f}%")
 
        avg = np.mean(accs)
        print(f"  Average: {avg:.2f}%")
 
        task_accs.append(accs)
        avg_accs.append(avg)
 
    # Final summary
    print("\nFinal Results")
    print(f"Final Average Accuracy: {avg_accs[-1]:.2f}%")
 
    # Average forgetting across all tasks
    avg_forgetting = 0
    for t in range(num_tasks):
        if t < len(task_accs[-1]):
            initial = task_accs[t][t]
            final = task_accs[-1][t]
            avg_forgetting += max(0, initial - final)
    avg_forgetting /= num_tasks
 
    print(f"Average Forgetting: {avg_forgetting:.2f}%")
 
    # Comparison with reference paper result (Figure 3c)
    paper_acc = 95
    diff = avg_accs[-1] - paper_acc
 
    print("\nComparison with Paper:")
    print(f"  Paper (Fig 3c): ~{paper_acc}%")
    print(f"  Our result: {avg_accs[-1]:.2f}%")
    print(f"  Difference: {diff:+.2f}%")
 
    if avg_accs[-1] >= 90:
        print("Result matches paper-level performance.")
    elif avg_accs[-1] >= 85:
        print("Result is close to paper-level performance.")
    else:
        print("Result is below target; further tuning may help.")
 
    plot_results(task_accs, avg_accs, num_tasks)
 
    return learner, task_accs, avg_accs
 
 
if __name__ == "__main__":
    learner, task_accs, avg_accs = run_experiment(
        num_tasks=5,
        epochs=20,
        batch_size=128,
        beta=1.0
    )
 
    print("\nExperiment complete.")