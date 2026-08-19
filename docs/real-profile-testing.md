# Real profile security checks

The default test suite uses committed SQLite exports and therefore cannot prove
properties of a report produced by the local Nsight runner. On a CUDA-capable
machine with `nsys` and `nvcc`, run the load-bearing capture check explicitly:

```bash
NSYS_REAL_CAPTURE=1 pytest tests/test_real_capture_security.py -v
```

The test compiles a tiny CUDA workload, runs it through `LocalProfileRunner`,
and verifies that the resulting capture does not persist the runner's declared
environment. CI records this test as an opt-in skip because hosted runners do
not provide Nsight Systems or a CUDA device; the fast argv assertion remains in
the ordinary suite.
