.PHONY: sync test lint typecheck check import audit manifests

UV = UV_CACHE_DIR=.uv-cache uv

sync:
	$(UV) sync --python 3.11

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
