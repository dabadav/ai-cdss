# Implementation

## Initial Prescriptions (First Week)

- **Filter Protocols**: Remove protocols with a lower PPF (Protocol Performance Factor) than a specified threshold.
- **Create Tuples**: Generate tuples of scores combining prognostics-difficulty-match and PPF.
- **Sort and Select**: Sort protocols and select the top 12.
- **Permutation**: Alternate protocols to ensure variety.

## Prognostics-Difficulty-Match (PDM)

- **Patient Class**: Classify patients based on recovery speed (slow, medium, fast).
- **Protocol Difficulty**: Assess protocol difficulty from the Protocol Matrix (low, medium, high).
- **PDM Calculation**: Compute the absolute difference between patient class and protocol difficulty.

## Update Prescriptions (Weekly)

- **Decision Making**: For each protocol, decide to keep it or perform a Type 1 swap.
- **Value Function**: Use the value function to determine protocol value:

Coefficients are initialized to 0.33. Adherence is calculated based on user login days.

- **Swapping Mechanism**: 
- Fetch the 5 most similar protocols.
- Swap with the most novel protocol (least number of sessions).
- Break ties by recency, choosing the least recent protocol.

### Type 2 Swaps (Optional)

- **Random Selection**: Pick `n` random prescriptions.
- **Exploration**: Swap with a dissimilar protocol to explore new options.