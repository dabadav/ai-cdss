
# Protocol Evaluation

We will score the protocols based on the following variables: 

- **Patient-Protocol Fit (PPF)**

Similarity between patient profile $P_u$ (normalized patient deficiency in clinical subscales) and protocol profile $Q_p$  (normalized protocol attributes, on the impact of a protocol to clinical subscales improvement) for all protocols.

$$PPF = \cos(\theta) = \frac{P \cdot Q}{\|P\|\|Q\|}$$

- **Recent Difficulty Modulator Changes ($\Delta DM$)**:

Exponentially weighted average increase in difficulty modulators across recent sessions, emphasizing recent performance.

$$\Delta DM_t = \frac{\sum_{i=0}^{t}(1 - \alpha)^i \cdot DM_{t - i}}{\sum_{i=0}^{t}(1 - \alpha)^i}$$

- **Recent Adherence**:

Session-wise adherence for all past sessions computed in minutes of protocol $u$ for a patient $p$.

$$\text{Adherence} = \frac{\text{Completed Duration}}{\text{Prescribed Duration}}$$

### Scoring Function

Each protocol's overall score combines these factors:

$$\text{Score} = \alpha \cdot PPF + \beta \cdot \Delta DM + \gamma \cdot \text{Adherence}$$

Coefficients $\alpha, \beta, \gamma$ weigh factor importance.