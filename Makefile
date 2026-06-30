.PHONY: all
all: out/summary.txt out/samples.png
	less out/summary.txt
	feh out/*.png

.PHONY: clean
clean:
	rm -rf build/
	rm -rf out/

cmdstan/bin/stanc:
	./scripts/install_cmdstan.sh

build/model.hpp: cmdstan/bin/stanc src/model.stan
	mkdir -p build/
	cp src/model.stan build/model.stan
	./cmdstan/bin/stanc -fsoa --O1 --warn-uninitialized --warn-pedantic build/model.stan

build/model: scripts/build_model.sh build/model.hpp
	./scripts/build_model.sh

out/data.json: scripts/export_data.py
	mkdir -p out/
	./scripts/export_data.py

out/samples.csv: build/model out/data.json
	./build/model num_threads=-1 random seed=123456789 sample num_chains=1 num_warmup=1000 num_samples=1000 data file=out/data.json output file=out/samples.csv

out/summary.txt: out/samples.csv
	./cmdstan/bin/stansummary out/samples.csv > out/summary.txt

out/samples.png: out/samples.csv ./scripts/plot_summary.py
	./scripts/plot_summary.py
