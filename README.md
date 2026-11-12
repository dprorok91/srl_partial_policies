# Deep and Symbolic Reinforcement Learning: From Policies to Heuristics

## Objective

The goal is to explore **Deep Reinforcement Learning (DRL) and Symbolic Reinforcement Learning (SRL)** through practical implementations, experiments, and increasingly complex environments.

While SRL can discover compact mathematical policies for relatively simple problems, its applicability to more complex environments is limited. We want to investigate whether **symbolic methods can instead discover useful partial strategies and heuristics**, which neural agents can selectively exploit.

Of particular interest is whether combining neural and symbolic approaches can **reduce computational costs without sacrificing performance**, while making some aspects of the learned strategies more interpretable.

The project is primarily an opportunity to learn and experiment with RL techniques, with potential research questions emerging along the way.

## Development Roadmap

**Stage 1 — Deep Reinforcement Learning**

Implement and train a neural RL agent (SAC) on Gymnasium `Pendulum-v1`. Understand the algorithm and establish performance and computational benchmarks.

**Stage 2 — Symbolic Reinforcement Learning**

Implement a standalone SRL agent that discovers mathematical control policies directly through interaction with the same environment.

**Stage 3 — Evaluation and Comparison**

Compare both approaches in terms of performance, sample efficiency, computational cost, robustness, and interpretability.

**Stage 4 — Symbolic Heuristics for Complex Problems**

Move to a more complex optimization game where a complete symbolic solution is impractical. Explore methods for automatically discovering useful local heuristics and subproblem strategies.

**Stage 5 — Hybrid Neuro-Symbolic RL**

Experiment with neural agents that selectively employ discovered symbolic heuristics. Evaluate whether this improves learning efficiency, computational cost, or generalization compared with standalone DRL.

## Long-Term Direction

Explore the possibility of combining **automatically discovered mathematical heuristics with flexible neural decision-making**, potentially extending symbolic RL to problems that cannot be efficiently described by a single symbolic policy.

Stages 1–3 form the initial hands-on learning project. Stages 4–5 are exploratory extensions, depending on what we learn from the earlier experiments.
