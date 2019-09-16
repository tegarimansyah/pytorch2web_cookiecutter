help:            ## Show this help.
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'
	
install:         ## Install virtualenv, requirements and install it on jupyter notebook
	[ -d ".venv" ] || virtualenv .venv
	@( \
		. .venv/bin/activate; \
		pip install -r requirements.txt; \
		ipython kernel install --user --name .venv --display-name "Pytorch {{cookiecutter.project_name}}"; \
	)

serve:           ## Run web server to predict
	uvicorn app.main:app --reload --port 8001

train:           ## Run training progress
	@( \
			. .venv/bin/activate; \
		pytest app/architecture/train.py; \
	)