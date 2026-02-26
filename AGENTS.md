# GUI Project Agent Guide

## Scope
- Repository: `streaming video understanding and GUI`
- Working directory: `/home/v-tangxin/GUI`
- Runtime environment: Azure VM with GPU

## Environment Baseline
- GPU: `NVIDIA A100 80GB PCIe`
- NVIDIA Driver: `580.95.05` (CUDA driver `13.0`)
- PyTorch CUDA: `12.8` (compatible with current driver)

## Session Bootstrap
1. `cd /home/v-tangxin/GUI`
2. `source ml_env/bin/activate`
3. Always use `python3` (never `python`)

## Preferred Workflow (Default)
1. Clarification phase:
- Use `interview-mode` to clarify goals, constraints, and assumptions.
- Use `brainstorming` when ideas are still open-ended.
2. Planning phase:
- Produce an executable plan with clear steps, dependencies, and checkpoints.
3. Implementation phase:
- Execute plan step by step.
- Keep changes scoped to task goals and acceptance metrics.
4. Review phase:
- Run `code-simplifier` for readability, consistency, bug/risk checks, and optimization opportunities.
5. Finalization phase:
- Validate outputs against success metrics.
- Prepare and push changes.

## Prompt Contract For Task Requests
When starting a task, user input should ideally include:
- Goal: what must be achieved
- Evaluation metrics: how success is measured
- Context: relevant files, constraints, prior decisions

If any of the three is missing for a non-trivial task, run `interview-mode` first.

## Implementation Rules
- Prioritize correctness first, then maintainability, then speed.
- Make incremental, testable changes.
- Avoid broad refactors unless explicitly requested.
- Keep plans and execution traceable to metrics.

## Common Commands
- Data generation:
  - `python3 -m agent.src.pipeline --compare --scenes general --limit 1 --verbose`
- Frontend preview:
  - `./ml_env/bin/python3 -m agent.preview.server --port 8000`

## Search Preference
- Prefer Tavily-enabled web search when external research is required.
