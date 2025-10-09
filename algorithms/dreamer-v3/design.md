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
- Replay buffer $D$ stores sequences $(x_t, a_t, r_t, c_t)$.
- Initialize world model $\phi$, actor $\theta$, critic $\psi$; initialize EMA critic $\bar{\psi} \leftarrow \psi$.
- Repeat until training ends:
  1. Collect experience
     - For each env step, sample action $a_t \sim \pi_{\theta}(\cdot\mid s_t)$ where $s_t$ is obtained by filtering with the posterior $q_{\phi}$ along the current trajectory.
     - Step the environment: $(x_{t+1}, r_t, \mathrm{done}) \leftarrow \mathrm{env.step}(a_t)$, set $c_t = \mathbb{1}[\neg\,\mathrm{done}]$, and append $(x_t, a_t, r_t, c_t)$ to $D$.
  2. Perform gradient updates (per train ratio)
     - Sample a batch of $B$ sequences of length $T$ from $D$.
     - World model update (filtering)
       - For $t=1,\dots,T$:
         $$\begin{aligned}
         e_t &= enc_{\phi}(x_t),\\
         h_t &= f_{\phi}(h_{t-1}, a_{t-1}, z_{t-1}),\\
         q_t &= q_{\phi}(z_t\mid h_t, x_t),\quad p_t = p_{\phi}(z_t\mid h_t),\\
         z_t &\sim q_t\ \text{(straight-through)},\quad s_t=(h_t, z_t),\\
         \hat{x}_t &= dec_{\phi}(s_t),\ \hat{r}_t = p_{\phi}(r_t\mid s_t),\ \hat{c}_t = p_{\phi}(c_t\mid s_t).
         \end{aligned}$$
       - Compute $\mathcal{L}_{\mathrm{world}}$ and update $\phi$.
     - Imagination (start states detached from grads)
       - Let $\bar{s}_0$ be posterior states from the batch (e.g., last time step), with stop-gradient.
       - For $\tau=0,\dots,H-1$:
         $$\begin{aligned}
         a_\tau &\sim \pi_{\theta}(\cdot\mid \bar{s}_\tau),\\
         \bar{h}_{\tau+1} &= f_{\phi}(\bar{h}_\tau, a_\tau, \bar{z}_\tau),\\
         \bar{z}_{\tau+1} &\sim p_{\phi}(z\mid \bar{h}_{\tau+1}),\\
         \bar{s}_{\tau+1} &= (\bar{h}_{\tau+1}, \bar{z}_{\tau+1}),\\
         r_\tau &= \operatorname{symexp}(\mathbb{E}[\hat{r}(\bar{s}_\tau)]),\quad c_\tau = \sigma(\hat{c}(\bar{s}_\tau)),\\
         v_{\tau+1} &= v_{\psi}(\bar{s}_{\tau+1}).
         \end{aligned}$$
       - Compute $\lambda$-returns in real space, with $d_\tau = \gamma\, c_\tau$:
         $$R_\tau^{\lambda} = r_\tau + d_\tau\Big((1-\lambda)\, v_{\tau+1} + \lambda\, R_{\tau+1}^{\lambda}\Big).$$
       - Return scaling: $S_{\mathrm{eff}} = \max\big(\operatorname{Per}(R^{\lambda}, P_{\text{high}}) - \operatorname{Per}(R^{\lambda}, P_{\text{low}}),\ 1\big)$.
     - Critic update (distributional two-hot on symlog targets)
       - Targets $y_\tau = \mathrm{twohot}(\operatorname{symlog}(R_\tau^{\lambda}))$.
       - Minimize $\mathcal{L}_{\mathrm{critic}}$ and regularize with EMA: $\mathcal{L}_{\mathrm{ema}}=\mathrm{KL}( p_{\bar{\psi}}\Vert p_{\psi})$; update $\psi$ and $\bar{\psi}$.
     - Actor update (no gradients through $\psi$)
       - Maximize $\mathbb{E}\big[ R_0^{\lambda} / S_{\mathrm{eff}} + \eta\, \mathcal{H}(\pi_{\theta}(\cdot\mid \bar{s}_\tau)) \big]$; update $\theta$.

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
