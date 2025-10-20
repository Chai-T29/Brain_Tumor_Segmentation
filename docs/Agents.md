# Agents Quick Reference

This cheat sheet keeps future Codex sessions aligned with the Brain Tumor Segmentation project setup.

## Environment Basics
- Shell: `powershell.exe`; prefer PowerShell-native commands, fall back to `python -c` or `bash -lc` only when required.
- Repo root: `D:\Brain_Tumor_Segmentation`.
- Sandbox: full filesystem access, network enabled; approval policy `never` (no escalation requests).
- Active branch: `suraj-continuous` unless changed explicitly.

## Project Structure
- `data/` – dataset utilities such as `dataset.py`, cached embeddings, and helper scripts.
- `docs/` – project notes (`guidance_updates.md`, this file, and other documentation).
- `dqn/` – legacy DQN implementation and related assets.
- `lightning_logs/` – PyTorch Lightning run artifacts (checkpoints, TensorBoard event files).
- `MU-Glioma-Post/` – original dataset directory (large medical imaging volumes).
- `rl/` – current reinforcement-learning stack (agent, environment, replay buffer, Lightning module).
- `scripts/` – CLI utilities or maintenance scripts.
- `tests/` – pytest suites covering dataset, n-step logic, TD3 agent, etc.
- Root-level helpers – configs (`base_config.yaml`, `config.yaml`), notebooks (`main.ipynb`), training/entry scripts (`train_dqn.py`, `test_dqn.py`), and repo metadata (`.gitignore`, `requirements.txt`).

## Shell & File Ops
- Always supply `workdir` when calling the shell tool; avoid `cd` inside commands.
- Prefer `rg`/`rg --files` for search; use other tools only if necessary.
- Default to ASCII when editing; preserve existing formatting and style unless a change is required.

## Git & Editing
- Never revert or overwrite user-created edits; stage only modifications you make intentionally.
- Reference files with inline code paths and optional line numbers (`rl/agent.py:240`).
- Run targeted tests after meaningful code changes; remove temporary scripts before finishing.

## Planning & Communication
- Create multi-step plans for anything beyond trivial edits.
- Keep replies concise, friendly, and action-focused; lead with the outcome when summarizing code changes.
- During reviews, highlight bugs/risks first, then optional improvements.

## Reinforcement-Learning Context
- Guidance system supports deterministic schedules, optional Beta randomization, guided actor loss, and replay targets (see `docs/guidance_updates.md` for a deep dive).
- `TD3Agent` logs `guidance_scale`; monitor this during runs to understand exploration pressure.

## Testing Reminders
- Typical command: `$env:PYTHONPATH='.'; pytest tests/<target>.py`.
- Always report which tests ran (or note if none were executed and why).

## Escalation Rules
- With approval policy `never`, solve issues within sandbox limits. If a command needs higher privileges, find another approach instead of requesting escalation.

## Stuff to Add
- Document standard training commands for the latest TD3 configuration (once stabilized).
- Capture baseline metrics (reward curves, IOU, loss trends) after rerunning deterministic and randomized guidance setups.
- Add notes on dataset preprocessing expectations (spacing, normalization) for new contributors.

Keep this file updated as workflows evolve so new sessions ramp up quickly.
