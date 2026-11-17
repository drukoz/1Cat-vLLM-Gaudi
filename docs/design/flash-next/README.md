# Flash-Next on Gaudi program (RUK-2709)

Program documents for serving Qwen3.8-Flash-Next on Intel Gaudi2 with this plugin fork. The working copies live on the Paperclip task RUK-2709; these files are refreshed whenever the pull request is updated.

- [System map](system-map.md): what the plugin is, the model and its memory budget on Gaudi2, the node and the two software stacks, the user's reference measurements on Qwen3.8-27B, upstream state, program shape and research status. Version 0.2 (2026-11-17) adds the user's answers and the R2 and R4 research results.
- [Reuse ledger](reuse-ledger.md): every capability the program needs and what already provides it, with the one vLLM engine pin (`8f816a3f`).
- Run sheets (`run-sheets/`, from M0): the exact commands, expected log lines and results tables the user executes on the node, since no agent has hardware access.
- Plan: written after the research subtasks and the first spikes report.
