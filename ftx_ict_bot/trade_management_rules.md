# ICT Trade Management Rules

This document outlines the context-specific rules for managing open positions.

## Guiding Principles
1.  **Protect Capital:** The first priority is always to move to a risk-free position.
2.  **Let Winners Run:** Once risk-free, the goal is to capture as much of the expected move as possible.
3.  **Respect the Narrative:** If the market narrative that justified the trade breaks down, the trade should be re-evaluated or closed.

## Core Management Strategies

### 1. Move to Breakeven (BE)
-   **Trigger:** When the trade achieves a 1:1 Risk-to-Reward (1R) profit.
-   **Action:** Move the Stop Loss to the entry price.
-   **Applies to:** All trade types (Reversal and Continuation).

### 2. Partial Profit Taking
-   **Strategy A (Liquidity-Based):**
    -   **Trigger:** When price hits a significant opposing liquidity level (e.g., an old high/low, a major FVG).
    -   **Action:** Close 50% of the position. Trail the Stop Loss on the remaining position.
    -   **Applies to:** Primarily for trades targeting specific HTF liquidity.

-   **Strategy B (Fixed R:R):**
    -   **Trigger:** When the trade achieves a 1:2 Risk-to-Reward (2R) profit.
    -   **Action:** Close 50% of the position. Move SL to Breakeven if not already done.
    -   **Applies to:** All trade types as a baseline.

### 3. Stop Loss Trailing
-   **Method:** Trail the Stop Loss behind confirmed market structure.
-   **Trigger (for Buys):** After a new Higher High (HH) is formed, trail the SL to the newly created Higher Low (HL).
-   **Trigger (for Sells):** After a new Lower Low (LL) is formed, trail the SL to the newly created Lower High (LH).
-   **Applies to:** The remaining position after partial profits have been taken.

### 4. Trade Invalidation (Narrative Failure)
-   **Trigger:** A confirmed Market Structure Shift (CHoCH) against the trade's direction.
    -   *Example:* If in a BUY trade, and a bearish CHoCH occurs, the bullish narrative is invalidated.
-   **Action:** Close the trade immediately at market.
-   **Applies to:** All open trades. This is a critical override to prevent a winning trade from turning into a loss.

---
*This is a foundational set of rules. We will expand upon this with more nuanced strategies for specific entry models (e.g., Unicorn vs. OB entries) as we build out the system.*
