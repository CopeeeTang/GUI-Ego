# Mixed Strategy Evaluation Report

> Date: 2026-02-20
> Cases: 60 (full), 15 (RT subset)
> Params: ANTICIPATION=1.0, LATENCY=2.0, COOLDOWN=12.0

## 1. Overall Results

| Strategy | avg F1 (60 cases) | Description |
|----------|------------------------|-------------|
| **Baseline** | 0.203 | gap>0, cd=12 |
| **A: Type-Optimal AD** | 0.237 | Per-type best tau from Phase II |
| **B: Mixed Routing** | 0.245 | RT/BL/AD routing by task type |

**Delta B vs A**: +0.009 (Mixed vs Type-Optimal AD)
**Delta B vs BL**: +0.042 (Mixed vs Baseline)

### RT Subset Comparison (15 cases)

| Strategy | avg F1 |
|----------|--------|
| Baseline | 0.225 |
| A: Type-Optimal AD | 0.298 |
| B: Mixed Routing | 0.332 |
| C: Oracle per-case | 0.361 |

## 2. Per-Type Breakdown

| Task Type | N | BL avg | A avg | B avg | B method | B-A delta | Optimal tau |
|-----------|---|--------|-------|-------|----------|-----------|-------------|
| Action Reasoning | 5 | 0.442 | 0.442 | 0.442 | AD_mid | +0.000 | 0.0 |
| Action Recognition | 5 | 0.208 | 0.212 | 0.212 | AD_mid | +0.000 | 0.5 |
| Attribute Perception | 5 | 0.216 | 0.256 | 0.256 | AD_mid | +0.000 | 0.5 |
| Ego Object Localization | 5 | 0.208 | 0.223 | 0.223 | AD_mid | +0.000 | 1.5 |
| Ego Object State Change Recognition | 5 | 0.097 | 0.199 | 0.199 | AD_mid | +0.000 | 0.5 |
| Information Function | 5 | 0.118 | 0.143 | 0.143 | AD_mid | +0.000 | 2.8 |
| Object Function | 5 | 0.164 | 0.231 | 0.231 | AD_mid | +0.000 | 0.5 |
| Object Localization | 5 | 0.061 | 0.173 | 0.173 | AD_mid | +0.000 | 5.0 |
| Object Recognition | 5 | 0.128 | 0.143 | 0.143 | AD_mid | +0.000 | 2.5 |
| Object State Change Recognition | 5 | 0.144 | 0.144 | 0.144 | AD_mid | +0.000 | 0.0 |
| Task Understanding | 5 | 0.388 | 0.388 | 0.388 | BL | +0.000 | 0.0 |
| Text-Rich Understanding | 5 | 0.265 | 0.288 | 0.390 | RT | +0.103 | 2.0 |

## 3. Strategy B Case-by-Case Impact

- **Improved**: 1 cases
- **Unchanged**: 59 cases
- **Worsened**: 0 cases

### Top Improvements (B > A)

| Case ID | Task Type | A F1 | B F1 | Delta | Method |
|---------|-----------|------|------|-------|--------|
| ec8c429db0d8 | Text-Rich Understanding | 0.154 | 0.667 | +0.513 | RT |

## 4. Oracle Analysis (RT Subset)

| Case ID | Task Type | RT F1 | AD F1 | BL F1 | Winner |
|---------|-----------|-------|-------|-------|--------|
| f7328dc0ae72 | Object Recognition | 0.167 | 0.211 | 0.211 | AD |
| 8563cf568b39 | Object State Change Recognition | 0.000 | 0.250 | 0.250 | AD |
| 1b83f0628411 | Ego Object State Change Recognition | 0.000 | 0.154 | 0.143 | AD |
| ec8c429db0d8 | Text-Rich Understanding | 0.667 | 0.154 | 0.154 | RT |
| 0eb97247e799 | Object Recognition | 0.250 | 0.077 | 0.077 | RT |
| 022907170077 | Object Localization | 0.000 | 0.200 | 0.200 | AD |
| eb1a417d26c9 | Object Recognition | 0.000 | 0.429 | 0.353 | AD |
| 85a99403e624 | Object State Change Recognition | 0.500 | 0.250 | 0.250 | RT |
| 4a51f9ebaa22 | Object Function | 0.133 | 0.471 | 0.333 | AD |
| a14e569060e8 | Object Function | 0.154 | 0.250 | 0.222 | AD |
| 2098c3e8d904 | Action Recognition | 0.125 | 0.250 | 0.235 | AD |
| 958a57945bfc | Ego Object State Change Recognition | 0.000 | 0.500 | 0.000 | AD |
| 5f334102e386 | Attribute Perception | 0.400 | 0.500 | 0.333 | AD |
| 48c5b16addef | Attribute Perception | 0.194 | 0.444 | 0.444 | AD |
| e58c97407823 | Object Function | 0.000 | 0.333 | 0.167 | AD |

**Oracle winner distribution**: {'AD': 12, 'RT': 3}

## 5. Routing Table Used

```python
# Explicit routing overrides:
#   Task Understanding: BL
#   Text-Rich Understanding: RT
# All other types: AD with type-optimal tau from Phase II sweep

# Effective tau per type:
#   Action Reasoning: AD (tau=0.0)
#   Action Recognition: AD (tau=0.5)
#   Attribute Perception: AD (tau=0.5)
#   Ego Object Localization: AD (tau=1.5)
#   Ego Object State Change Recognition: AD (tau=0.5)
#   Information Function: AD (tau=2.8)
#   Object Function: AD (tau=0.5)
#   Object Localization: AD (tau=5.0)
#   Object Recognition: AD (tau=2.5)
#   Object State Change Recognition: AD (tau=0.0)
#   Task Understanding: BL (tau=0.0)
#   Text-Rich Understanding: RT (tau=2.0)
```

## 6. Conclusions

Mixed routing (B) improves over Type-Optimal AD (A) by **+0.009** F1.

Both improve over Baseline: A=0.237, B=0.245 vs BL=0.203

Oracle upper bound on 15-case RT subset: 0.361 (A=0.298, B=0.332)
Gap to oracle: 0.028 — room for improvement if RT coverage expands.
