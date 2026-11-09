.PHONY: clean install-mmseqs

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
TOOLS_DIR := $(PROJECT_DIR)/tools
MMSEQS_VERSION := 15-6f452
MMSEQS_DIR := $(TOOLS_DIR)/mmseqs

PYTHON_INTERPRETER=python
ifeq (,$(shell which poetry))
HAS_POETRY=False
else
HAS_POETRY=True
endif


#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Initialize environment with poetry, pre-commit, and mmseqs2
init:

ifeq (True,$(HAS_POETRY))
	poetry install
	poetry run pre-commit install
	poetry run pre-commit install --hook-type commit-msg
else
	@echo ">>>No poetry detected, not installing pre-commit and packages"
endif
	@echo ">>>Installing MMseqs2..."
	$(MAKE) install-mmseqs

## Install MMseqs2 for sequence clustering and homology detection
install-mmseqs:
	@echo "Installing MMseqs2 to $(MMSEQS_DIR)..."
	@mkdir -p $(TOOLS_DIR)
	@if [ -f "$(MMSEQS_DIR)/bin/mmseqs" ]; then \
		echo "MMseqs2 already installed at $(MMSEQS_DIR)"; \
		$(MMSEQS_DIR)/bin/mmseqs version; \
	else \
		cd $(TOOLS_DIR) && \
		wget -q https://github.com/soedinglab/MMseqs2/releases/download/$(MMSEQS_VERSION)/mmseqs-linux-avx2.tar.gz && \
		tar xzf mmseqs-linux-avx2.tar.gz && \
		rm mmseqs-linux-avx2.tar.gz && \
		echo "MMseqs2 installed successfully at $(MMSEQS_DIR)/bin/mmseqs"; \
		$(MMSEQS_DIR)/bin/mmseqs version; \
	fi
	@echo "Use --mmseqs_bin=$(MMSEQS_DIR)/bin/mmseqs or add to PATH"

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
