# Flash-Next on Gaudi program (RUK-2709)

Program documents for serving Qwen3.8-Flash-Next on Intel Gaudi2 with this plugin fork. The working copies live on the Paperclip task RUK-2709; these files are refreshed whenever the pull request is updated.

- [System map](system-map.md): what the plugin is, the model and its memory budget on Gaudi2, the node and the two software stacks, the user's reference measurements on Qwen3.8-27B, upstream state, program shape and research status. Version 0.5 (2026-11-17) closes the research phase: all research and spike results are in, section 17 lists the facts the plan rests on.
- [Reuse ledger](reuse-ledger.md): every capability the program needs and what already provides it, with the one vLLM engine pin (`8f816a3f`).
- [Plan](plan.md): target architecture (upstream model subclassed with three HPU seams), the CPU oracle (Hugging Face transformers on a tiny model), conventions, milestones M0–M5 with their subtasks, the qualification run sheet, the spike table and the checklist. Version 0.1 (2026-11-17).
- Run sheets (`run-sheets/`, from M0): the exact commands, expected log lines and results tables the user executes on the node, since no agent has hardware access.
