# LocalAnonPipeline: quick start (Linux, macOS). On Windows use run.ps1 (see README.md).
#
#   make run IN=/path/to/export                     # output: out/<export name> in this folder
#   make run IN=/path/to/export OUT=/path/to/output
#
# IN is an export folder or an archive (.zip, .tar.gz). The first start sets everything up (README.md).

.DEFAULT_GOAL := help
.PHONY: help run

help:
	@echo "usage: make run IN=<export folder or archive> [OUT=<output folder>]"
	@echo "       the output goes to out/<export name> in this folder by default; details: README.md"

run:
	@if [ -z "$(IN)" ]; then echo "usage: make run IN=<export folder or archive> [OUT=<output folder>]" >&2; exit 2; fi
	bash ./run.sh "$(IN)" $(if $(OUT),"$(OUT)",)
