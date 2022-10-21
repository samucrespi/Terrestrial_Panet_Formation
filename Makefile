
# --- Help
help:
	@echo
	@echo 'Supported Makefile invocations:'
	@echo 'pall	- runs all the SPH_Rebound_*.py in parallel'
	@echo

# --- pall
pall:
	parallel -j10 < parallel.run
