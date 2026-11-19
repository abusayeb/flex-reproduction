# Per-operation error rates, LLM `ollama:qwen2.5:14b-instruct`, T=0.0

| Operation | Errors | Rate | Paper (GPT-4o) |
|---|---|---|---|
| add | 12/21 | 57.1% | 8.6% |
| generalize | 13/21 | 61.9% | 5.7% |
| mask | 12/21 | 57.1% | 3.8% |

Contradictory Add replies: 12

Most common errors (op, scenario, got, expected): count
- ('add', 'Add Only', ('goal',), ('goal', 'stove')): 3
- ('add', 'Add+Gen.', ('fridge', 'goal'), ('fridge', 'goal', 'stove')): 3
- ('add', 'Add+Mask', ('fridge', 'goal'), ('fridge', 'goal', 'stove')): 3
- ('add', 'All Three', ('fridge', 'goal'), ('fridge', 'goal', 'stove')): 3
- ('generalize', 'Mask Only', ('fridge', 'goal', 'heat_source'), ('fridge', 'goal', 'stove')): 3
- ('mask', 'Gen. Only', ('goal',), ('goal', 'stove')): 3
- ('mask', 'Mask Only', ('goal',), ('goal', 'stove')): 3
- ('mask', 'Add+Gen.', ('goal',), ('fridge', 'goal')): 3
- ('mask', 'Mask+Gen.', ('goal',), ('goal', 'stove')): 3
- ('generalize', 'Add+Gen.', ('goal', 'icy'), ('fridge', 'goal')): 2
- ('generalize', 'Mask+Gen.', ('goal', 'heat_source', 'light_source'), ('fridge', 'goal', 'heat_source')): 2
- ('generalize', 'All Three', ('goal', 'icy'), ('fridge', 'goal')): 2
