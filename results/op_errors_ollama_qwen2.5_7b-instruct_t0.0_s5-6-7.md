# Per-operation error rates, LLM `ollama:qwen2.5:7b-instruct`, T=0.0

| Operation | Errors | Rate | Paper (GPT-4o) |
|---|---|---|---|
| add | 12/21 | 57.1% | 8.6% |
| generalize | 6/21 | 28.6% | 5.7% |
| mask | 12/21 | 57.1% | 3.8% |

Contradictory Add replies: 12

Most common errors (op, scenario, got, expected): count
- ('add', 'Add Only', ('goal',), ('goal', 'stove')): 3
- ('add', 'Add+Gen.', ('fridge', 'goal'), ('fridge', 'goal', 'stove')): 3
- ('add', 'Add+Mask', ('fridge', 'goal'), ('fridge', 'goal', 'stove')): 3
- ('add', 'All Three', ('fridge', 'goal'), ('fridge', 'goal', 'stove')): 3
- ('generalize', 'Gen. Only', ('goal', 'stove'), ('goal', 'heat_source')): 3
- ('generalize', 'Mask+Gen.', ('fridge', 'goal', 'stove'), ('fridge', 'goal', 'heat_source')): 3
- ('mask', 'Gen. Only', ('goal',), ('goal', 'stove')): 3
- ('mask', 'Mask Only', ('goal',), ('goal', 'stove')): 3
- ('mask', 'Add+Gen.', ('goal',), ('fridge', 'goal')): 3
- ('mask', 'Mask+Gen.', ('goal',), ('goal', 'stove')): 3
