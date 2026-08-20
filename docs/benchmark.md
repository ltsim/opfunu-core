# Benchmark Base Classes

Every function in `opfunu-core` inherits from one of the base classes defined in the `opfunu.benchmark` package:

| Class            | Description                                                  | Used by                       |
|------------------|--------------------------------------------------------------|-------------------------------|
| `Benchmark`      | Abstract base class defining the common function interface   | all functions                 |
| `Formula`        | Dataclass holding a LaTeX formula (rendered by IPython)      | all functions                 |
| `FuncBenchmark`  | Base class for traditional (name-based) functions            | `opfunu.name_based.*`         |
| `CecBenchmark`   | Base class for CEC competition functions                     | `opfunu.cec_based.*`          |

## Benchmark and Formula

::: opfunu.benchmark
    options:
      members: [Formula, Benchmark]

## FuncBenchmark

::: opfunu.benchmark.func.FuncBenchmark

## CecBenchmark

::: opfunu.benchmark.cec.CecBenchmark
