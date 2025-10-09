# DreamerV3: Design Document

**Main Idea**
- DreamerV3 learns a latent world model from experience and trains its actor and critic purely from imagined trajectories in the learned latent space. The world model encodes observations into discrete stochastic representations and predicts their evolution under actions via a recurrent sequence model. The critic estimates returns for imagined futures and the actor is optimized to select actions that lead to high imagined returns, with robust normalization and transformations (notably symlog) that allow a single configuration to work across diverse domains.

**Components, Variables, Parameters**
- Variables
  - $x_t$ observation; $a_t$ action; $r_t$ reward; $c_t\in\{0,1\}$ continue (1 if episode not terminated); $\gamma$ discount; $\lambda$ TD-lambda; $H$ imagination horizon.
  - Deterministic state $h_t$; stochastic discrete latent $z_t$; combined latent state $s_t=(h_t, z_t)$.
  - Prior over latents $p_{\phi}(z_t\mid h_t)$; posterior $q_{\phi}(z_t\mid h_t, x_t)$; decoder $p_{\phi}(x_t\mid s_t)$; reward head $p_{\phi}(r_t\mid s_t)$; continue head $p_{\phi}(c_t\mid s_t)$.
  - Actor $\pi_{\theta}(a_t\mid s_t)$; critic value distribution $p_{\psi}(b\mid s_t)$ over bins $b\in\mathbb{B}$.
  - Stop-gradient operator $\mathrm{sg}[\,\cdot\,]$.
- World Model (parameters $\phi$)
  - Encoder $enc_{\phi}: x_t\to e_t$ (embedding).
  - Sequence model $f_{\phi}: (h_{t-1}, a_{t-1}, z_{t-1})\to h_t$ (GRU-based RSSM core).
  - Dynamics predictor (prior) $p_{\phi}(z_t\mid h_t)$; Representation (posterior) $q_{\phi}(z_t\mid h_t, x_t)$; $z_t$ is discrete with $G$ groups, $K$ classes per group (straight-through Gumbel-Softmax for training).
  - Decoder $dec_{\phi}: s_t\to \hat{x}_t$ for input reconstruction.
  - Reward predictor $p_{\phi}(r_t\mid s_t)$; Continue predictor $p_{\phi}(c_t\mid s_t)$.
- Actor–Critic (parameters $\theta, \psi$)
  - Actor policy $\pi_{\theta}(a\mid s)$ (Tanh-Normal for continuous, Categorical for discrete).
  - Critic $p_{\psi}(b\mid s)$ predicts a discrete distribution over symlog value bins; value readout $v_{\psi}(s)=\mathbb{E}_{b\sim p_{\psi}}[\operatorname{symexp}(b)]$.
- Transforms
  - $\operatorname{symlog}(x)=\mathrm{sign}(x)\cdot\log(1+|x|)$; $\operatorname{symexp}(y)=\mathrm{sign}(y)\cdot(\exp(|y|)-1)$.
  - Two-hot discretization: map scalar target $y$ (in symlog space) to adjacent bin weights $y_{\mathrm{vec}}$ over bins $\mathbb{B}$ for discrete regression.
- Major symbols (concrete defaults below)
  - Discount $\gamma$; TD-lambda $\lambda$; imagination horizon $H$.
  - KL balancing factor $\alpha$ ($0<\alpha<1$); free-bits threshold $\delta_{\mathrm{fb}}$ (per-latent minimal KL); KL scale $\beta_{\mathrm{kl}}$.
  - Value/reward bins $N_{\mathrm{bins}}$; bin range in symlog space $[b_{\min}, b_{\max}]$.
  - Actor entropy scale $\eta$; return scaling percentiles $P_{\text{low}}$, $P_{\text{high}}$.
  - Critic EMA decay $\rho$ and EMA regularizer weight $w_{\mathrm{ema}}$.
  - Training ratio $\rho_{\text{train}}$ (updates per env step); batch size $B$; sequence length $T$.
  - Optimizer: learning rate(s) and Adam $\beta$s.
  - Discrete latent sizes: groups $G$, classes $K$, Gumbel temperature schedule $\tau$.

**Hyperparameters (DreamerV3 defaults)**
- Training schedule
  - Train ratio $\rho_{\text{train}}$: 32 (varies by suite; e.g., 256–1024 for 100k/1M budgets)
  - Batch size $B$: 16; sequence length $T$: 64; report length: 32
- Discounting and horizons
  - Continue discounting enabled ($\text{contdisc}=\text{True}$); $\gamma = 1 - 1/H$ with $H = 333$ $\Rightarrow$ $\gamma \approx 0.997$
  - TD-$\lambda$: $\lambda = 0.95$ (for imagination and replay targets)
  - Imagination length: 15
- Return scaling (fixed-entropy trick)
  - Percentiles: $P_{\text{low}}=5$, $P_{\text{high}}=95$; $S_{\mathrm{eff}}=\max(\operatorname{Per}(R^{\lambda},95)-\operatorname{Per}(R^{\lambda},5), 1)$
  - Actor entropy scale $\eta = 3\cdot 10^{-4}$
- Critic stability
  - EMA/slow value rate: $0.02$ ($\approx$ decay $\rho = 0.98$ applied every step); regularizer weight $w_{\mathrm{ema}} = 1.0$
  - Value head bins $N_{\mathrm{bins}} = 255$ (symlog two-hot)
- Reward and continue heads
  - Reward bins $N_{\mathrm{bins}} = 255$ (symlog two-hot); Continue head: Bernoulli
- World model (RSSM, discrete latents)
  - Discrete: stoch=32, classes=64 (G=32 groups, K=64 classes)
  - Deterministic size deter=8192; hidden=1024; blocks=8; unimix=0.01; free_nats=1.0
  - Prior/obs MLP layers: imglayers=2, obslayers=1, dynlayers=1; act=SiLU; norm=RMS
  - KL balancing via loss scales: $\text{dyn}=1.0$, $\text{rep}=0.1$ (equivalently $\alpha\approx0.9$ toward prior)
- Encoder/Decoder
  - Encoder: depth=64, mults=[2,3,4,4], layers=3, units=1024, act=SiLU, norm=RMS, kernel=5, symlog=True
  - Decoder: depth=64, mults=[2,3,4,4], layers=3, units=1024, act=SiLU, norm=RMS, kernel=5, bspace=8
- Actor/Critic networks
  - Policy: layers=3, units=1024, act=SiLU, minstd=0.1, maxstd=1.0, outscale=0.01, unimix=0.01
  - Critic: layers=3, units=1024, act=SiLU, bins=$255$, output=$\text{symexp\_twohot}$
- Optimizer
  - Adam: $\mathrm{lr}=4\cdot 10^{-5}$, $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-20}$, global grad clipping (AGC)$=0.3$, weight decay$=0.0$
- Loss weights
  - rec=1.0, rew=1.0, con=1.0, dyn=1.0, rep=0.1, policy=1.0, value=1.0, repval=0.3

**Losses**
- World model losses (optimize $\phi$)
  - $\mathcal{L}_{\mathrm{pred}}$ (Prediction): shapes encoder/decoder and supervised heads
    - Image reconstruction $\mathcal{L}_{\mathrm{img}}$: $\mathbb{E}_{t}\big[\,\lVert \operatorname{symlog}(x_t) - \operatorname{symlog}(\hat{x}_t) \rVert_2^2\,\big]$ where $\hat{x}_t = dec_{\phi}(s_t)$.
    - Reward prediction $\mathcal{L}_{\mathrm{rew}}$: two-hot cross-entropy on symlog reward $\hat{y}_t=\operatorname{symlog}(r_t)$, $\mathcal{L}_{\mathrm{rew}}=\mathbb{E}_{t}[\, \mathrm{CE}(\mathrm{twohot}(\hat{y}_t),\ p_{\phi}(\cdot\mid s_t))\,]$.
    - Continue prediction $\mathcal{L}_{\mathrm{cont}}$: binary cross-entropy $\mathbb{E}_{t}[\, \mathrm{BCE}(c_t,\ p_{\phi}(c_t=1\mid s_t))\,]$.
  - $\mathcal{L}_{\mathrm{dyn}}$ (Dynamics / KL with balancing + free bits): encourages prior to match posterior
    - $\mathrm{KL}_{\text{post}\to\text{prior}}=\mathrm{KL}\big(q_{\phi}(z_t\mid h_t,x_t)\,\Vert\, p_{\phi}(z_t\mid h_t)\big)$
    - $\mathrm{KL}_{\text{prior}\to\text{post}}=\mathrm{KL}\big(p_{\phi}(z_t\mid h_t)\,\Vert\, q_{\phi}(z_t\mid h_t,x_t)\big)$
    - Balanced, stop-grad: $\mathcal{L}_{\mathrm{dyn}}= \alpha\,\mathrm{KL}(\mathrm{sg}[q] \Vert p) + (1-\alpha)\,\mathrm{KL}(q \Vert \mathrm{sg}[p])$, with free-bits per latent group $\max(\mathrm{KL}-\delta_{\mathrm{fb}}, 0)$.
  - World model objective: $\mathcal{L}_{\mathrm{world}}= w_{\mathrm{img}}\,\mathcal{L}_{\mathrm{img}} + w_{\mathrm{rew}}\,\mathcal{L}_{\mathrm{rew}} + w_{\mathrm{cont}}\,\mathcal{L}_{\mathrm{cont}} + \beta_{\mathrm{kl}}\,\mathcal{L}_{\mathrm{dyn}}$.
- Critic losses (optimize $\psi$)
  - Targets from imagined rollouts (see loop): compute $\lambda$-returns in real space using predicted rewards and continues, then transform to symlog and discretize to two-hot vector $y_t$.
  - $\mathcal{L}_{\mathrm{critic}}$ (distributional regression): $\mathbb{E}_{t}[\, \mathrm{CE}( y_t,\ p_{\psi}(\cdot\mid s_t) )\,]$.
  - EMA regularizer: $\mathcal{L}_{\mathrm{ema}}=\mathbb{E}_{t}[\, \mathrm{KL}( p_{\bar{\psi}}(\cdot\mid s_t)\,\Vert\, p_{\psi}(\cdot\mid s_t) )\,]$, with $\bar{\psi} \leftarrow \rho\, \bar{\psi} + (1-\rho)\, \psi$.
  - Critic objective: $\mathcal{L}_{\mathrm{val}}=\mathcal{L}_{\mathrm{critic}} + w_{\mathrm{ema}}\, \mathcal{L}_{\mathrm{ema}}$.
- Actor loss (optimize $\theta$)
  - Imagine with current policy; compute $\lambda$-returns $R_t^{\lambda}$ in real space: $R_t^{\lambda}= r_t + d_t\,\big[(1-\lambda)\, v_{\psi}(s_{t+1}) + \lambda\, R_{t+1}^{\lambda}\big]$, where $d_t = \gamma\, c_t^{\mathrm{pred}}$ and $v_{\psi}$ uses $\operatorname{symexp}$ on the critic distribution; stop-grad through $v_{\psi}$ when computing targets.
  - Scale returns for stability without amplifying noise: let $S = \operatorname{Per}(R^{\lambda}, P_{\text{high}}) - \operatorname{Per}(R^{\lambda}, P_{\text{low}})$ over the imagination batch; use $S_{\mathrm{eff}}=\max(S, 1)$.
  - Entropy-regularized objective: maximize $\mathbb{E}\big[ R_0^{\lambda} / S_{\mathrm{eff}} + \eta\, \mathcal{H}(\pi_{\theta}(\cdot\mid s_t)) \big]$.
  - Implement as minimization: $\mathcal{L}_{\mathrm{actor}}= -\mathbb{E}\big[ R_0^{\lambda} / S_{\mathrm{eff}} + \eta\, \mathcal{H}(\pi_{\theta}(\cdot\mid s_t)) \big]$.

**Training Loop (Pseudo Code)**
```python
# Replay D stores (x_t, a_t, r_t, c_t) with sequences
initialize world model φ, actor θ, critic ψ; EMA critic ψ_bar = ψ
while not done:
  # 1) Collect experience
  for env_step in range(E):
    with torch.no_grad():
      # use latest policy; optionally warmstart with random actions
      a_t ~ π_θ(a | s_t) where s_t from filtering posterior q_φ on current traj
    (x_{t+1}, r_t, done) = env.step(a_t); c_t = 0 if done else 1
    D.append(x_t, a_t, r_t, c_t)

  # 2) Gradient updates (ρ_train * E updates)
  for update in range(U):
    batch = sample_sequences(D, batch_size=B, length=T)

    # 2.a) World model update
    # Encode and posterior-filter through time
    h_0 = zeros(); z_0 = zeros()
    for t in 1..T:
      e_t = enc_φ(x_t)
      h_t = f_φ(h_{t-1}, a_{t-1}, z_{t-1})
      q_t = q_φ(z_t | h_t, x_t)        # posterior
      p_t = p_φ(z_t | h_t)             # prior
      z_t ~ q_t                        # posterior sample (straight-through)
      s_t = (h_t, z_t)
      x̂_t = dec_φ(s_t); r̂_t = head_r_φ(s_t); ĉ_t = head_c_φ(s_t)
      accumulate L_img, L_rew, L_cont, L_dyn (balanced KL with free bits)
    L_world = w_img*L_img + w_rew*L_rew + w_cont*L_cont + β_kl*L_dyn
    φ ← φ - η_φ * ∇_φ L_world

    # 2.b) Imagine trajectories and compute targets
    # Start from posterior states detached from gradients
    s̄_0 = sg[s_T_end_of_chunks]
    returns = []
    with torch.no_grad():
      R = 0
    for t in 0..H-1:
      a_t ~ π_θ(a | s̄_t)
      h̄_{t+1} = f_φ(h̄_t, a_t, z̄_t)
      z̄_{t+1} ~ p_φ(z | h̄_{t+1})         # prior during imagination
      s̄_{t+1} = (h̄_{t+1}, z̄_{t+1})
      r_t = symexp(E[r̂(s̄_t)])            # reward head expectation
      c_t = sigmoid(head_c_φ(s̄_t))        # continue prob
      v_{t+1} = v_ψ(s̄_{t+1})              # symexp of critic dist
      # accumulate λ-returns backward later
    R^λ = lambda_returns(r, c, v, γ, λ)     # real-space returns
    S_eff = max(percentile(R^λ, P_high) - percentile(R^λ, P_low), 1)

    # 2.c) Critic update (distributional two-hot on symlog targets)
    y_t = twohot(symlog(R^λ_t), bins=𝔹)
    L_critic = CE(y_t, p_ψ(·|s̄_t)); L_ema = KL(p_{ψ_bar}(·|s̄_t) || p_ψ(·|s̄_t))
    ψ ← ψ - η_ψ * ∇_ψ (L_critic + w_ema*L_ema)
    ψ_bar ← ρ * ψ_bar + (1-ρ) * ψ

    # 2.d) Actor update (reparameterized gradients through π, no gradients through ψ)
    J = mean(R^λ_0 / S_eff + η * entropy(π_θ(·|s̄_{0:H-1})))
    θ ← θ + η_θ * ∇_θ J  # or minimize L_actor = -J
```

**Network Architectures**
- Encoder $enc_{\phi}$
  - Visual inputs: CNN with 4 conv blocks, stride-2 downsampling to $1/16$ spatial size; channels e.g., $[32, 64, 128, 256]$; kernel $4$ or $3$; activation SiLU/ReLU; flatten + linear to embedding dim $E$.
  - Low-dim inputs: MLP 2–3 layers, hidden $256$–$512$, SiLU/ReLU to embedding $E$.
- RSSM core $f_{\phi}$
  - Input: concat $[z_{t-1}, a_{t-1}] \to$ linear $\to$ GRU ($H_{\mathrm{det}}$ units) $\to h_t$.
  - Prior head: MLP($h_t$) $\to$ logits of discrete latents with shape $(G, K)$.
  - Posterior head: MLP([$h_t$, $e_t$]) $\to$ logits of discrete latents $(G, K)$; sample with straight-through Gumbel-Softmax (temperature $\tau$).
- Decoder $dec_{\phi}$
  - MLP to reshape, then 4 deconvs mirroring encoder; output image mean $\hat{x}_t$; optimize $\ell_2$ in symlog space.
- Reward head $p_{\phi}(r\mid s)$
  - MLP 2–3 layers; output logits over $N_{\mathrm{bins}}$ bins in symlog space; trained via two-hot CE.
- Continue head $p_{\phi}(c\mid s)$
  - MLP 2–3 layers; output Bernoulli prob; BCE loss.
- Critic $p_{\psi}(b\mid s)$
  - MLP 3 layers (hidden $512$–$1024$); output logits over $N_{\mathrm{bins}}$; trained by CE to two-hot of symlog $\lambda$-returns; EMA target for regularization.
- Actor $\pi_{\theta}(a\mid s)$
  - MLP 3 layers (hidden $512$–$1024$); outputs:
    - Continuous: mean and log-std per action dim; Tanh squashing; reparameterized sampling.
    - Discrete: logits over actions; straight-through gradient for sampling where needed.

Notes
- Use the same heads for imagination as for reconstruction but disable teacher forcing: during imagination, use prior $p_{\phi}(z\mid h)$ and policy actions; stop-grad through critic targets when training the actor.
- Discount uses predicted continue: $d_t = \gamma \cdot c_t$; for terminal transitions, $c_t=0$.
- All world-model and critic predictions for rewards/values use symlog; actor objective uses real-space returns scaled by $S_{\mathrm{eff}}$ to keep a single entropy scale $\eta$ across tasks.

File ownership
- This document guides the implementation under `algorithms/dreamer-v3/` and should be kept in sync with code once added.
