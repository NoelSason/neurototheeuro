# neurototheeuro — Agent Context

Research repo for a self-supervised **brain foundation model** — can one model learn a transferable "language of the brain" that decodes new brains with little/no retraining? See `README.md` for the full thesis and the Model A vs. Model B de-risking experiment.

**Current state:** vision + a staged dataset (`ds005273/`, BIDS metadata only — the raw EEG signal binaries still need to be fetched via DataLad). No analysis code or results yet. This is an independent research track; keep it self-contained.

## Model Orchestration Workflow

Work on this repo follows a shared model-orchestration hierarchy. Every agent MUST know its role and stay in it.

**Roles**
- **Fable 5 (high effort) — Chief Planner / Architect.** Writes the plan, sets research + architecture direction, breaks work into assigned subtasks; owns writing and design-from-scratch. Plans and delegates; does not grind out large implementations.
- **Opus 4.8 — Lead Implementer & Decision-Maker.** Implements the plan and makes the on-the-ground engineering decisions for its assigned domain; delegates small, well-scoped pieces to Sonnet 5 but stays accountable.
- **Sonnet 5 — Scoped Executor.** Small, clearly-defined subtasks handed down by Opus 4.8 (mechanical edits, localized fixes, boilerplate).
- **GPT-5.6-Sol (Codex) — Diligent Implementation Engine.** The parallel workhorse for long-running, grind-heavy work: data fetching/preprocessing, training runs, robustness and test coverage, multi-day experiments.

**Flow:** Fable 5 plans → assigns subtasks → Opus 4.8 implements & decides (delegating small work to Sonnet 5), while GPT-5.6-Sol runs assigned data/experiment/robustness tracks in parallel.

**Assignment enforcement (CRITICAL):** Before starting ANY task, check the model assigned to it in the plan/tracker. If a task is NOT assigned to you, STOP WORK IMMEDIATELY. Do not proceed. Tell the user: "This task is assigned to [Model], not me — please run it with the correct model."

**Source of truth for who-does-what:** this repo's own plan/tracker.
