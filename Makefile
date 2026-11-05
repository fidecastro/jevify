# Shortcuts for the two everyday jevify commands: `make ask` asks typed questions
# about one state and prints the answer as JSON; `make serve` serves Jev's HTTP API
# for one recipe. Values reach jevify through the environment, so a state may hold
# spaces, quotes and apostrophes. `make help` lists the variables and the recipes.

JEVIFY ?= uv run jevify
HOST   ?= 127.0.0.1
PORT   ?= 8600

export RECIPE STATE CHOICE SCORE NOUL IMAGE RUNG HOST PORT

.DEFAULT_GOAL := help
.PHONY: help ask serve

help:
	@echo "make ask   RECIPE=<file> STATE=<text|@file> [CHOICE=...] [SCORE=...] [NOUL=...] [IMAGE=<png|jpg|webp>] [RUNG=<rung>]"
	@echo "make serve RECIPE=<file> [HOST=$(HOST)] [PORT=$(PORT)]"
	@echo ""
	@echo "  CHOICE  id:question=option1,option2,...   one option from a menu"
	@echo "  SCORE   id:question=level0,level1,...     a position on ordered levels, low to high"
	@echo "  NOUL    id:statement                      the probability that the statement is true"
	@echo "  Give at least one of CHOICE, SCORE, NOUL. For several of one type, call jevify ask directly."
	@echo "  make reads \$$ in a value: write \$$\$$49.99 for \$$49.99, or put the text in a file and pass STATE=@path."
	@echo "  HOST other than 127.0.0.1 needs JEVIFY_API_KEY set: clients must then send it as a bearer key."
	@echo ""
	@echo "Example:"
	@echo "  make ask RECIPE=recipes/gemma-4-e4b-it.llamacpp.yaml \\"
	@echo "    STATE=\"Customer: I was charged twice. Fix it today or I cancel.\" \\"
	@echo "    CHOICE=\"dept:Which team should handle this?=billing,support,sales\" \\"
	@echo "    SCORE=\"urgency:How urgent is this?=low,medium,high\" \\"
	@echo "    NOUL=\"cancel:The customer threatens to cancel.\""
	@echo ""
	@echo "Recipes:"
	@ls recipes/*.yaml | sed 's/^/  /'

ask:
	@test -n "$$RECIPE" || { echo "make ask: set RECIPE=recipes/<name>.yaml (make help lists them)" >&2; exit 2; }
	@test -n "$$STATE" || { echo "make ask: set STATE=\"...\" (the text, or @path to a file)" >&2; exit 2; }
	@set -- --state "$$STATE"; \
	[ -z "$$CHOICE" ] || set -- "$$@" --choice "$$CHOICE"; \
	[ -z "$$SCORE" ] || set -- "$$@" --score "$$SCORE"; \
	[ -z "$$NOUL" ] || set -- "$$@" --noul "$$NOUL"; \
	[ -z "$$IMAGE" ] || set -- "$$@" --image "$$IMAGE"; \
	[ -z "$$RUNG" ] || set -- "$$@" --rung "$$RUNG"; \
	$(JEVIFY) ask "$$RECIPE" "$$@"

serve:
	@test -n "$$RECIPE" || { echo "make serve: set RECIPE=recipes/<name>.yaml (make help lists them)" >&2; exit 2; }
	@case "$$HOST" in 127.0.0.1|localhost|::1) ;; *) test -n "$$JEVIFY_API_KEY" || { echo "make serve: HOST=$$HOST opens the API to the network; export JEVIFY_API_KEY first" >&2; exit 2; } ;; esac
	$(JEVIFY) serve "$$RECIPE" --host "$$HOST" --port "$$PORT"
