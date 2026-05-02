# Executive Summary: Estonian Energy Resilience Under Disruption

**Advanced Business Analytics Final Project | May 2026**

---

## The Problem

In January 2026, Estonia faced an acute energy crisis. Geopolitical disruption cut Russian gas imports, and unusually calm weather (wind 50% below average) reduced renewable generation. The country currently depends on imports for **40–60% of electricity demand**. 

**Key question:** Can strategic wind farm investments restore energy independence, and what is the residual risk?

---

## Our Solution

We developed an **AI-powered forecasting system** that combines:
- **Machine learning** (Spatio-Temporal Graph Neural Network) to predict energy supply under different scenarios
- **Physics-based wind production models** (ERA5 weather + Vestas turbine curves) for planned wind farms
- **Risk analysis** (worst-case vs. average outcomes) to identify true vulnerabilities

---

## Key Findings

| Scenario | Wind Capacity Added | Mean Supply Impact | Deficit Hours Reduction | Verdict |
|----------|---|---|---|---|
| **Current (S1)** | — | +300 MW (with imports) | 0 | Baseline: imports provide cushion |
| **Full Isolation (S2)** | — | −200 MW | 744 hours | Catastrophic: cannot self-supply |
| **Scenario A** | +323 MW | −130 MW | 40% reduction | Significant progress; **not sufficient** |
| **Scenario B** | +887 MW | −100 MW | 65% reduction | Largely solves problem; **requires all farms** |

### Critical Insight
**Weather variability dominates outcomes.** January 2026 wind was 50% below normal. Even Scenario B (1,581 MW total capacity) leaves ~250–350 deficit hours during calm weather. **Wind alone is insufficient for true resilience.**

---

## Policy Recommendations

### ✅ For Grid Operators (Elering)

1. **Prioritize Scenario A wind farms** (Lääneranna, Pärnu, Aidu)
   - 323 MW new capacity; 2–3 year timeline
   - Reduces crisis hours by 40%
   - High permit certainty

2. **Accelerate Scenario B deployment** (5 pipeline municipalities)
   - 564 MW additional; 4–5 year timeline
   - Achieves energy independence for 80% of hours
   - Requires multi-municipal coordination

3. **Implement supplementary measures** (critical)
   - **Strategic reserves:** 3–6 months gas/hydro storage (€300–500M)
   - **Demand flexibility:** 30% of load shiftable via industrial contracts + EV incentives (€50–100M)
   - **Energy storage:** 200–500 MW battery capacity (€100–300M)

### 🎯 Bottom Line
- **Investment required:** ~€1 billion for full wind expansion + reserves
- **Payoff:** Eliminates €200–300M annual emergency import premiums
- **Timeline:** 4–5 years to full resilience

---

## What Makes This Analysis Credible

✓ **Data-driven:** Real grid data (2019–2026) + actual January 2026 weather  
✓ **Realistic:** Wind modeled from physics (Vestas turbines, air density corrections)  
✓ **Risk-aware:** Evaluates worst-case hours (P10 quantile), not just averages  
✓ **Tested:** Model trained on 7 years; tested on unseen 2026 crisis month  
✓ **Actionable:** Specific farms, capacities, deployment sequencing  

---

## Limitations

- Analysis assumes single weather point (spatial variations ignored)
- Wind farms modeled in isolation (grid integration losses not included)
- Assumes ideal dispatch (real system more constrained)
- Does not model political/regulatory implementation barriers

---

## Implementation Timeline

| **2026–2028** | Build Scenario A (323 MW) → 40% resilience improvement |
|---|---|
| **2028–2031** | Build Scenario B (564 MW) → Energy independence achieved |
| **2026+** | Deploy AI forecasting system → Daily 24h-ahead supply forecasts |
| **2026–2030** | Build strategic reserves + demand flexibility infrastructure |

---

**Supporting materials:** Technical report (Final_Project.ipynb), code & reproducible pipeline  
**Contact:** [Team email]  
**Date:** 25 April 2026
