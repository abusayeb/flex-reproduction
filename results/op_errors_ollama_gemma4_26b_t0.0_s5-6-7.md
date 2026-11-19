# Per-operation error rates, LLM `ollama:gemma4:26b`, T=0.0

| Operation | Errors | Rate | Paper (GPT-4o) |
|---|---|---|---|
| add | 0/21 | 0.0% | 8.6% |
| generalize | 3/21 | 14.3% | 5.7% |
| mask | 3/21 | 14.3% | 3.8% |

Contradictory Add replies: 0

Most common errors (op, scenario, got, expected): count
- ('generalize', 'Mask Only', ('fridge', 'goal', 'heat_source'), ('fridge', 'goal', 'stove')): 3
- ('mask', 'Add+Gen.', ('goal',), ('fridge', 'goal')): 3
