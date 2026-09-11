# Robust Pre-training under Imperfect Information: Replay Imitation and Opponent-Belief Search

### Pokémon TCG AI Battle Challenge — Strategy Category

## Overview and Hypotheses

The public baseline (`reinforcement-learning-and-mcts-sample-code.ipynb`) pairs a Transformer-style value–policy network — an encoder–decoder built from multi-head attention and position-wise feed-forward blocks, following Vaswani et al. (2017) — with Monte-Carlo Tree Search (MCTS). It is a strong scaffold, but its *learning signal* is narrow: the network is trained purely by self-play on a **single fixed deck** (a cold start with no external data), and during search the opponent's hidden cards are filled with **dummy cards in one arbitrary way**. Yet the Pokémon TCG is an *imperfect-information* game: the opponent's hand and deck order are unobserved, and outcomes depend heavily on the luck of the opening hand and draws.

We therefore redesigned the pre-training stage around three hypotheses.

- **H1 (central).** If the opponent's hidden state is instantiated from an *estimated distribution* rather than a dummy, and search is run over several such instantiations (determinization particles), the agent depends less on any particular initial state or draw and plays more consistently across repeated matches.
- **H2.** Learning from a large corpus of *real battle replays* with *advantage-weighted imitation* (AWR) yields a stronger and more consistent policy than uniform imitation of every played move.
- **H3.** Assigning credit with a *per-game adaptive discounted return*, rather than pasting the terminal ±1 onto every position, stabilizes the value estimate.

## Training Data

We replaced self-play with pre-training on the **full public daily-replay corpus**. Daily datasets were retrieved from the episode index and stored under `episodes/` (`download_daily_dataset.py`, recursive zip reading).

- **Scale:** 46 days (2026-07-01 to 08-15), **218,366 games** in total (4,340–5,333 per day).
- **Rationale:** real replays contain diverse opponents, the live metagame, and realistic matchups — a far richer teacher than a cold-start self-play loop on one deck.

Real decks were grouped into **16 archetype clusters** by hierarchical clustering (similarity threshold 0.75). For each cluster we extracted two disjoint sample sets: **own** (moves made by the player piloting that archetype) and **opp** (moves made by players facing that archetype). Each position is converted to a sparse feature vector and written to shards (`preprocess_shards.py`).

The number of extracted decision samples is highly skewed across archetypes (**Table 1**). This skew is strategically decisive: data-rich archetypes such as `cl00` can be trained well, whereas rare decks (`cl15` has a few hundred positions) cannot be learned in isolation. We therefore **focused the final submission on the most data-dense archetype, `cl00`**.

**Table 1. Extracted decision samples per archetype (own, representative).**

| Cluster | own samples | | Cluster | own samples |
|---|--:|---|---|--:|
| cl00 | 5,897,867 | | cl08 | 380,291 |
| cl01 | 4,191,551 | | cl10 | 153,675 |
| cl02 | 1,288,681 | | cl12 | 18,632 |
| cl04 | 1,003,201 | | cl14 | 4,010 |
| cl05 | 921,341 | | cl15 | 655 |

*(≈15.3M samples across 16 clusters; a spread of roughly four orders of magnitude.)*

## Learning Logic: Differences from the Baseline

**Table 2. Baseline vs. our pre-training.**

| Aspect | Baseline | Ours |
|---|---|---|
| Training data | Self-play on one deck | Public replays, 46 days / 218k games |
| Deck coverage | One fixed deck | Per-archetype self/opp (16 clusters) |
| Policy target | Self-play MCTS Q-advantage regression | **AWR** on real moves (advantage-weighted) |
| Policy head / prior | tanh output → `exp(policy×10)` | **tanh removed + temperature recalibrated (×1)** |
| Value target | TD(λ=0.9) bootstrap | **Per-game adaptive discounted return** (first move ≈0.3) |
| Opponent hidden cards | **Dummy fill (single guess)** | **Candidate-DB deck estimate + hand tracking + learned model, K=3 particles** |

### 3.1 Central contribution: belief-based determinization search

At the root of each search the baseline fills the opponent's deck with one card (Snorlax) and the hand and prizes with basic Energy. This single, non-existent, always-identical guess reflects neither the opponent's real threats nor the true draw distribution.

We instead treat the opponent's hidden state as an *estimated distribution* and instantiate it as follows.

1. **Deck estimation.** Observed opponent cards are matched against a candidate-deck database (738 real decks) to recover the most consistent real deck (`deck_reconstructor`).
2. **Known-hand tracking.** Card-movement logs are replayed to retain cards provably still in hand after searches and draws (`OpponentHandTracker`).
3. **Sampling unknowns.** From the remainder, hidden hand, deck and prizes are drawn while a **learned hand-retention model** reweights which cards the opponent is likely to be holding.
4. **Multi-particle search.** This yields **K=3 distinct fully-determined states (determinization particles)**; MCTS is run on each and the root visit counts are aggregated to choose the move.

This design *marginalizes* the imperfect information explicitly and directly targets the judging criterion of avoiding over-reliance on specific initial states, matchups, or situational advantages. Rather than the move that looks best under one lucky assumption, the agent selects the move that is **good on average across plausible boards**, making it robust to the opponent's draws and its own opening hand.

### 3.2 Replay imitation with AWR

The baseline's policy target is a regression of self-play MCTS Q-advantages. We instead learn from the **moves actually played** in replays, weighted by outcome: from the advantage `A = G_t − V(s)` (with the value head as baseline) we form `weight = exp(A_norm/β)`, so **moves on winning trajectories are reinforced and losing moves are almost ignored** — offline advantage-weighted regression. This avoids the failure mode of naive behavioral cloning, which reinforces the bad moves of lost games as strongly as good ones.

**Technical soundness (a design fix).** The baseline applies `tanh` to the policy head and forms the prior as `exp(policy×10)` at inference. That temperature suits the small regression targets of self-play, but with *imitation-trained* weights the prior **saturates**, collapsing search to one or two moves and nullifying the MCTS exploration term. We **removed the policy-head `tanh` and recalibrated the temperature (×1)**. Applying both transforms to the same model's raw logits over 8 games (911 multi-choice positions), the **baseline transform collapses 26.9% of positions to a single move (prior > 0.99), versus 10.1% for ours** (**Figure 2**), confirming that exploration now reaches the full action set.

### 3.3 Per-game adaptive discounted return

The baseline value target is a TD(λ=0.9) bootstrap of the model's own root value. We instead use a discounted return from the terminal result, `G_t = r · x^d`, with the discount rate set **per game** as `x = 0.3^{1/(N-1)}`. Independent of game length N, this normalizes the target so that the **first move is always ≈0.3 and the final move is ±1** (**Figure 1**). It avoids over-confidently pasting ±1 onto even early, balanced positions, and instead sharpens conviction toward the endgame. Empirically the value head stays well spread — mean −0.01, **std 0.47**, range [−0.97, +1.00] — rather than saturating at ±1 (**Figure 3**).

### 3.4 Per-archetype self/opp dual models

In the search tree, our own turns are evaluated by the **self** model and the **opponent's turns by a separate opp model** trained on how opponents play against that archetype. The belief system of §3.1 supplies *which board* the opponent might hold; the opp model supplies *how the opponent would act on it* — a two-layer treatment of the adversary that yields more realistic matchup-aware search.

## Experiments and Evidence

All measurements use the trained `cl00` model (8 games, 1,301 positions; values are direct measurements, not estimates).

- **Figure 1 — discounted-return curve.** First move 0.30, midgame ≈0.55, final move 1.00; identical for N = 40/80/120, confirming length-invariant credit assignment.
- **Figure 2 — prior collapse.** Baseline-style transform 26.9% → ours 10.1%: temperature/tanh recalibration cuts search collapse to roughly one third.
- **Figure 3 — value distribution.** Mean −0.01, std 0.47, range [−0.97, +1.00]: no ±1 saturation, resolution preserved.
- **External validation.** This pipeline reached roughly the **top 10%** of the Simulation Category.

![Figure 1. Per-game adaptive discounted return.](figures/fig_discount_curve.png)
![Figure 2. Policy-prior collapse rate.](figures/fig_prior_saturation.png)
![Figure 3. Value-head distribution.](figures/fig_value_dist.png)

## Deck Selection

The archetype clustering in Table 1 doubles as a deck decision: because pre-training quality tracks data volume, we submitted the archetype (`cl00`) with the deepest, most representative replay support, so that both the self policy and the opponent model rest on the largest evidence base and generalize across the matchups actually seen in the metagame.

## Conclusion

We extended the pre-training stage from the baseline's "single deck, self-play, dummy opponent" to **advantage-weighted imitation of real replays**, a **per-game adaptive discounted return**, and, most importantly, **belief-based multi-particle search** that marginalizes the opponent's hidden information. By confronting the imperfect-information nature of the Pokémon TCG directly, the agent reduces its dependence on specific initial states and draws and plays more consistently — with the temperature and discounting effects verified by direct measurement.
