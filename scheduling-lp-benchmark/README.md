# scheduling-lp-benchmark

Minimal benchmark harness for comparing LP solvers on scheduling instances.

Commands:

Generate instances:

```bash
python generate_instances.py --out instances
```

Run benchmark:

```bash
python benchmark.py --instances instances --out results/results.csv
```
