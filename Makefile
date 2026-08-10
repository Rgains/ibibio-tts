.PHONY: sync train-sync train-f5-sync demo-sync validate test lint typecheck check import audit manifests smoke f5-smoke full demo

UV = UV_CACHE_DIR=.uv-cache uv

sync:
	$(UV) sync --python 3.11

train-sync:
	$(UV) sync --python 3.11 --group train

train-f5-sync:
	$(UV) sync --python 3.11 --group train-f5

demo-sync:
	$(UV) sync --python 3.11 --group train --group demo

validate:
	$(UV) run python scripts/validate_environment.py

test:
	$(UV) run pytest

lint:
	$(UV) run ruff check .

typecheck:
	$(UV) run mypy src

check: test lint typecheck

import:
	$(UV) run python scripts/import_common_voice.py

audit:
	$(UV) run python scripts/audit_audio.py
	$(UV) run python scripts/analyse_speakers.py

manifests:
	$(UV) run python scripts/build_manifests.py

smoke:
	$(UV) run python scripts/train_vits_smoke.py

full:
	$(UV) run python scripts/train_vits_full.py

f5-smoke:
	$(UV) run python scripts/train_f5_smoke.py

demo:
	$(UV) run streamlit run app.py
