# Final-session forecast (written before the session)

The forecast itself is in [docs/forecast_final_session.md](../../../../docs/forecast_final_session.md). This folder holds what produced it:

- `scripts/forecast.py`: a deterministic simulation of 12 attempts per arm from the 20/24/25 Nov trial CSVs, the run logs and the 25 Nov camera results. It reads only files that are already in git.
- `results/forecast.json`, `results/forecast_tables.md`: the inputs, the per-arm priors and the forecast.
- `logs/forecast.sha256`, `logs/forecast_hashed_at.txt`: the hash and time written before the session.
  - They cover the forecast document, this folder's script and results, and the protocol draft as it was at forecast time.
  - Check with `sha256sum -c` from the repository root.

The forecast is not edited after hashing; it is scored against the session results (forecast section 5).
