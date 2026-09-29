# parry33 status

Last updated 2026-09-24.

- **Where it stopped:** the ML work paused on 2026-08-28. Working state, tooling and every retracted finding are in `docs/phase4-learned-audio.md`. Read it before resuming.
- **Repository:** `anfhann/parry33`, MIT-licensed, first pushed 2026-09-24. Reviewed for public release 2026-09-29: git history carries no game media (170 KiB total, no `.npy`/`.wav`/`.jpg`/`.pkl` ever committed), no secrets, and nothing machine-specific outside the gitignored `config/local.toml`.
- **Recorded capture removed 2026-09-24:** the 1,376 frame, clip, strip and audio files under `runs/` (6.49 GB). `runs/` keeps only the small per-session `events.jsonl` and `meta.json`, the `live_*` and `trig_*` logs, three trained models (`detector.pkl`, `grunt.pkl`, `grunt_keypress_backup.pkl`) and diagnostic PNGs, all outside git. New data means recording new fights.
- **Next action (code):** press timing. Detection is solved (~97-100% true recall); presses land at the wrong instant. The anchor was wrong twice — 553 ms before impact was too early, 180 ms was essentially the moment of contact and too late. Build up confirmed hits against misses with fixed-`--lead-ms` armed runs, turns skipped, and fit the offset from those.
- **This update:** Claude Code (Opus 5.5) in the Claude desktop app, during the Project Portage desktop sweep. No code changed.
