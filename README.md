# FLEX reproduction + FLEX-V

**Paper PDF:** [Read the full paper (arXiv:2610.28893)](https://arxiv.org/pdf/2610.28893).

This is my code for reproducing the FLEX paper, *Flexible Language-Guided Representation Editing for Adaptive Robot Behavior* (Lee, Dennler, Sucholutsky, Bobu, CoRL 2026, arXiv 2610.28893). I have done the 2D navigation part only (KitchenWorld and LavaLand).

I have also added one change of my own, which I am calling FLEX-V. In FLEX-V, every Generalize edit is checked against the demonstrations first, and the user is asked only when the demonstrations are not enough to decide.

The report is sent separately as a PDF.

## Running the code

Python 3.10 or newer is needed. I have used Python 3.11.

Cloning the repo, create and activate a virtual environment, then install the dependencies:

```bash
git clone https://github.com/abusayeb/flex-reproduction.git
cd flex-reproduction
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

The requirements are only numpy, scipy, matplotlib and pytest. `main.py` takes around one and a half minutes on my laptop.

You do not need Ollama or any API key. The `oracle` backend acts like a perfect LLM, and all the replies I got from the real models (Gemma 4 26B, Qwen 2.5 7B and 14B) are saved in `cache/`, so those runs just read from the cache.

`main.py` runs everything in this order:

- Part 0 (`basics`): checks the IRL alone with fixed features, no LLM involved
- Part 1 (`table1`): Table 1 of the report
- Part 2 (`op_errors`): error rate of each operation (Figure 2)
- Part 3 (`table2`): Table 2, FLEX vs FLEX-V
- Part 4 (`figures`): saves the three figures into `report/`

For every number it prints `ok` if it matches the report, otherwise `MISMATCH` along with the value in the report. If all is fine, the last lines say `All numbers match the report.`

To run only one part, give its name, for example:

```bash
python main.py table1
```

To run the unit tests:

```bash
python -m pytest
```

## Where the outputs go

- Table 1: `results/rq1_oracle_t0.0_d10.md` and `.json`
- Figure 2: `results/op_errors_*`
- Table 2: `results/ext_*`
- Figures: `report/fig_*.png`

The `.md` files have the summary tables. The `.json` files have the full details of every run (which features were chosen, the learned weights and the complete LLM log).

## Files

```text
flex/
  config.py       constants (grid size, reward weights, delta) and paths
  world.py        objects, corrections, random layouts
  features.py     features written as Python code, computed for every cell
  irl.py          planning and MaxEnt IRL
  prompts.py      prompts copied as it is from the paper
  llm.py          Ollama / OpenAI clients with caching, and the oracle
  editing.py      Add, Generalize, Mask, FLEX and the baselines
  flexv.py        my change (FLEX-V)
  scenarios.py    the 7 RQ1 scenarios and the personal preference ones
  experiment.py   runs one experiment and makes the result tables
  viz.py          drawing the layouts
main.py           runs everything
scripts/          run_rq1, run_op_errors, run_ext, make_figures
tests/            unit tests
cache/            saved LLM replies
results/          outputs of the experiments
```

## Things I had to decide myself

Some details are not given in the paper, so I have chosen these:

- The start is in the top-left 3x3 block, the goal is in the bottom-right 3x3 block, and the objects are placed on different cells of the middle 5x5 area. The object properties in LavaLand are my own.
- True reward = -1 x (distance to goal) + 8 x (distance to the hazard, capped at 3).
- Every path has exactly 17 states and has to end at the goal. Because of this, planning and IRL can be done exactly.
- In scenarios where Generalize is not needed, I have removed the second hazard from the layout.
- ALGAE is taken as only the Add step of FLEX (no contrastive demonstrations).
- If the feature code from the LLM crashes or is not additive, I use a template feature for the same property and count it as a fallback.

## Not done yet

- Full Table 1 with a real LLM
- Running GPT-4o itself
- RQ2 on the Franka arm
