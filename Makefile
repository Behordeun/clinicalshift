.PHONY: data vectorstores smoke run metrics verify clean all install

# uv-managed project — use `uv run` to invoke entry points
RUN := uv run
N_SMOKE := 10
N_FULL := 2000
SEED := 42

install:
	uv sync
	@echo "=== Dependencies installed ==="

data:
	$(RUN) generate-patients --n 50000 --seed $(SEED)
	$(RUN) generate-guidelines

vectorstores:
	$(RUN) build-vectorstores

verify:
	@echo "=== Verifying retrieval quality ==="
	$(RUN) python -c "import chromadb; c=chromadb.PersistentClient(path='data/chroma_db'); \
	coll=c.get_collection('tau_old'); r=coll.query(query_texts=['metformin eGFR CKD contraindication'],n_results=3); \
	ids=r['ids'][0]; print('tau_old top-3:', ids); assert 't2dm_02' in ids or 't2dm_04' in ids, 'FAIL: critical doc not in top-3'"
	$(RUN) python -c "import chromadb; c=chromadb.PersistentClient(path='data/chroma_db'); \
	coll=c.get_collection('instB'); r=coll.query(query_texts=['metformin eGFR CKD contraindication'],n_results=3); \
	ids=r['ids'][0]; print('instB top-3:', ids); assert 't2dm_02' not in ids and 't2dm_04' not in ids, 'FAIL: vocab shift not working'"
	@echo "=== Verification passed ==="

smoke:
	@echo "=== Smoke test: $(N_SMOKE) patients x all conditions ==="
	$(RUN) run-pipeline --run-all --n $(N_SMOKE) --seed $(SEED)
	$(RUN) compute-metrics
	@echo "=== Smoke test complete. Inspect results/ ==="

run:
	@echo "=== Full run: $(N_FULL) patients x all conditions ==="
	$(RUN) run-pipeline --run-all --n $(N_FULL) --seed $(SEED)

metrics:
	$(RUN) compute-metrics

clean:
	rm -f results/results_*.csv results/metrics_summary.csv
	rm -f results/plot_*.png results/plot_*.html

all: install data vectorstores verify run metrics
	@echo "=== Full pipeline complete ==="
