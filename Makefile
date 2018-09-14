PYTHON = python3
SETUP  := $(PYTHON) setup.py

.PHONY: clean venv

clean:
	$(SETUP) clean
	rm -rf .tox .eggs *.egg-info build dist venv
	@find . -regex '.*\(__pycache__\|\.py[co]\)' -delete

venv:
	$(PYTHON) -m virtualenv -p /usr/bin/$(PYTHON) venv
	@echo "Now run the following to activate the virtual env:"
	@echo ". venv/bin/activate"
