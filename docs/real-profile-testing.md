# Real profile security checks

The default test suite uses committed SQLite exports and therefore cannot prove
properties of a report produced by the local Nsight runner. On a CUDA-capable
machine with `nsys` and `nvcc`, the load-bearing capture check runs automatically.
Disable it explicitly when needed:

```bash
NSYS_REAL_CAPTURE=0 pytest tests/test_real_capture_security.py -v
```

The test compiles a tiny CUDA workload, runs it through `LocalProfileRunner`,
and verifies that the resulting capture does not persist the runner's declared
environment. Machines without `nsys` or `nvcc` skip it with the registered
`requires nsys and nvcc` reason; hosted CI therefore skips cleanly without
requiring a second opt-in flag. The fast argv assertion remains in the ordinary
suite.

## DistCA timeline profile

The timeline regression tests use the downloaded DistCA profile when it is
present at its example path. To run them against a copy stored elsewhere, set
`NSYS_DISTCA_SQLITE`:

```bash
NSYS_DISTCA_SQLITE=/path/to/megatron_distca.sqlite \
  pytest tests/test_timeline_web_distca_profile.py \
         tests/test_timeline_web_distca_benchmark.py -v
```

The same variable is honoured by
`examples/example-20-megatron-distca/benchmark_timeline_web.py`.
