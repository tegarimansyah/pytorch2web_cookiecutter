install:
	[ -d ".venv" ] || virtualenv .venv
	@( \
		. .venv/bin/activate; \
		pip install -r requirements.txt; \
		ipython kernel install --user --name=pytorch-{{cookiecutter.project_name}}; \
	)

serve:
	uvicorn app.main:app --reload --port 8001