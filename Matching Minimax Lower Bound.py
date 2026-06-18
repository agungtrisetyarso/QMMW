# ============================================================
# Colab Simulation of Theorem 2 — Matching Minimax Lower Bound
# Paper: Quantum Matrix Multiplicative Weights (Trisetyarso, 2026)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# ====================== HELPER FUNCTIONS ======================

def generate_rademacher_losses(d, T, seed=None):
    """Generate the exact hard instances from the proof of Theorem 2."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.choice([-1.0, 1.0], size=(T, d))

def run_hedge_on_rademacher(d, T, eta=None, losses=None, seed=42):
    """Run Hedge/MMW on Rademacher instances (reduces to classical case)."""
    if eta is None:
        eta = np.sqrt(2 * np.log(d) / T)

    if losses is None:
        losses = generate_rademacher_losses(d, T, seed=seed)

    w = np.ones(d) / d
    total_alg_cost = 0.0
    instantaneous_costs = []

    for t in range(T):
        p_t = w / np.sum(w)
        cost_t = np.dot(losses[t], p_t)
        instantaneous_costs.append(cost_t)
        total_alg_cost += cost_t
        w = w * np.exp(-eta * losses[t])

    cumulative_losses = np.sum(losses, axis=0)
    best_expert_cost = np.min(cumulative_losses)
    regret = total_alg_cost - best_expert_cost
    bound = np.sqrt(2 * T * np.log(d))

    gains = -losses
    max_walk = np.max(np.sum(gains, axis=0))

    return {
        'regret': regret,
        'bound': bound,
        'max_random_walk': max_walk,
        'instantaneous_costs': np.array(instantaneous_costs),
        'best_expert_cost': best_expert_cost
    }

def simulate_theorem2(d=64, T=2000, num_trials=60, seed=42):
    print("=" * 75)
    print("SIMULATION OF THEOREM 2 — Matching Minimax Lower Bound")
    print("Paper: Quantum Matrix Multiplicative Weights (Trisetyarso, June 2026)")
    print("=" * 75)
    print(f"Hard instance: Rademacher diagonal losses (as in proof of Theorem 2)")
    print(f"Parameters: d = {d}, T = {T}")
    print(f"Theoretical lower bound ≈ sqrt(2T log d) = {np.sqrt(2*T*np.log(d)):.3f}")
    print(f"Number of independent trials: {num_trials}\n")

    regrets = []
    for trial in tqdm(range(num_trials), desc="Simulating Rademacher instances"):
        res = run_hedge_on_rademacher(d=d, T=T, seed=seed + trial*777)
        regrets.append(res['regret'])

    regrets = np.array(regrets)
    bound = np.sqrt(2 * T * np.log(d))

    print("\n" + "-" * 50)
    print("RESULTS")
    print("-" * 50)
    print(f"Mean regret over {num_trials} trials : {np.mean(regrets):.3f} ± {np.std(regrets):.3f}")
    print(f"Theoretical lower bound (Theorem 2)  : {bound:.3f}")
    print(f"Ratio (mean regret / lower bound)    : {np.mean(regrets)/bound:.4f}")

    # One detailed run for plots
    print("\nRunning detailed visualization trial...")
    detailed = run_hedge_on_rademacher(d=d, T=T, seed=seed)

    # ====================== PLOTS ======================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    t_axis = np.arange(1, T + 1)
    cum_regret = np.cumsum(detailed['instantaneous_costs']) - detailed['best_expert_cost']

    # Plot 1: Cumulative regret on hard instance
    axes[0, 0].plot(t_axis, cum_regret, 'b-', linewidth=1.5, label='Hedge / MMW regret')
    axes[0, 0].axhline(y=bound, color='red', linestyle='--', linewidth=2, label=f'Theorem 2 lower bound ≈ {bound:.1f}')
    axes[0, 0].fill_between(t_axis, 0, cum_regret, alpha=0.15)
    axes[0, 0].set_xlabel('Round t')
    axes[0, 0].set_ylabel('Cumulative Regret')
    axes[0, 0].set_title(f'Cumulative Regret on Rademacher Hard Instance\n(d={d}, T={T})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Histogram of final regret
    axes[0, 1].hist(regrets, bins=20, color='coral', edgecolor='black', alpha=0.75)
    axes[0, 1].axvline(x=bound, color='red', linestyle='--', linewidth=2.5, label=f'Lower bound = {bound:.2f}')
    axes[0, 1].axvline(x=np.mean(regrets), color='darkgreen', linestyle='-', linewidth=2.5,
                       label=f'Mean = {np.mean(regrets):.2f}')
    axes[0, 1].set_xlabel('Final Regret')
    axes[0, 1].set_ylabel('Number of trials')
    axes[0, 1].set_title(f'Distribution of Regret over {num_trials} Hard Instances')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Ratio to lower bound
    ratio = regrets / bound
    axes[1, 0].hist(ratio, bins=18, color='lightgreen', edgecolor='black', alpha=0.75)
    axes[1, 0].axvline(x=1.0, color='red', linestyle='--', linewidth=2.5, label='Lower bound ratio = 1')
    axes[1, 0].axvline(x=np.mean(ratio), color='darkgreen', linestyle='-', linewidth=2.5,
                       label=f'Mean ratio = {np.mean(ratio):.4f}')
    axes[1, 0].set_xlabel('Regret / √(2T log d)')
    axes[1, 0].set_ylabel('Number of trials')
    axes[1, 0].set_title('Closeness to the Information-Theoretic Lower Bound')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Regret vs Max Random Walk (proof connection)
    max_walks = [run_hedge_on_rademacher(d=d, T=T, seed=seed + i*777)['max_random_walk'] 
                 for i in range(30)]
    regrets_scatter = [run_hedge_on_rademacher(d=d, T=T, seed=seed + i*777)['regret'] 
                       for i in range(30)]
    axes[1, 1].scatter(max_walks, regrets_scatter, alpha=0.6, s=50, c='purple', edgecolors='k')
    axes[1, 1].plot([min(max_walks), max(max_walks)], [min(max_walks), max(max_walks)],
                    'k--', linewidth=1.5, label='y = x')
    axes[1, 1].set_xlabel('Max Random Walk max_i S_T(i)')
    axes[1, 1].set_ylabel('Achieved Regret')
    axes[1, 1].set_title('Regret = max Random Walk (as predicted by proof)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Simulation of Theorem 2 — Minimax Lower Bound via Rademacher Instances', fontsize=14, y=1.02)
    plt.tight_layout()

    os.makedirs('/content', exist_ok=True)
    plot_path = '/content/theorem2_rademacher_simulation.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    plt.show()

    print("\n" + "=" * 75)
    print("KEY TAKEAWAYS FROM THEOREM 2 SIMULATION")
    print("=" * 75)
    print("• Even the optimal algorithm (MMW/Hedge) cannot beat the lower bound")
    print("  by more than lower-order terms on these hard instances.")
    print(f"• Achieved ratio ≈ {np.mean(ratio):.4f} (approaches 1 as T and d grow).")
    print("• Remark 1: These instances are commuting → classical algorithm matches MMW.")
    print("• Therefore Quantum MMW offers no regret advantage on the instances")
    print("  that certify optimality (only resource advantage: exponential space).")
    print("=" * 75)

    return detailed, regrets, bound

# ====================== RUN ======================
D = 64          # dimension (number of experts)
T = 2000        # time horizon
NUM_TRIALS = 60

detailed_result, all_regrets, theoretical_bound = simulate_theorem2(
    d=D, T=T, num_trials=NUM_TRIALS, seed=42
)
