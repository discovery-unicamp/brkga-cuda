# BrkgaCuda

A generic framework implementing the Biased Random-Key Genetic Algorithm
(BRKGA) for GPUs using CUDA. It aims to ease the development of BRKGA-based
solutions by encapsulating its core operations.

## Dependencies

This framework depends of:
- [bb_segsort](https://github.com/vtsynergy/bb_segsort.git)
- [CMake](https://cmake.org/)
- [CUDA Toolkit](https://docs.nvidia.com/cuda/index.html)
- [curand](https://developer.nvidia.com/curand)
- [OpenMP](https://www.openmp.org/)
    (optional; used for parallelization of CPU decoders)

## Usage

There are some examples (for evaluation purposes)
[here](https://github.com/abrunoaa-ic/brkga-cuda-experiments/tree/master/applications/src/brkga-cuda-2.0).
The following sample illustrates how to use the framework for the Traveling
Salesman Problem (TSP) using only the CPU to decode.

### Decoder

```C++
// TspDecoder.h
#include <brkga-cuda/Decoder.hpp>
#include <cmath>
#include <utility>
#include <vector>

class TspDecoder : public box::Decoder {
private:
  std::vector<float> distances;  // "matrix" of 2d distances

public:
  // required since we override just one overload
  using box::Decoder::decode;

  TspDecoder(const std::vector<std::pair<float, float>>& points)
      : distances(points.size() * points.size(), 0.0f) {
    const auto n = points.size();
    for (unsigned i = 0; i < n; ++i)
      for (unsigned j = i + 1; j < n; ++j)
        distances[i * n + j] = distances[j * n + i] =
            std::hypotf(points[i].first - points[j].first,
                        points[i].second - points[j].second);
  }

  // must implement a deterministic decoder for permutations
  float decode(const box::Chromosome<unsigned>& tour) const override {
    const auto n = config->chromosomeLength();  // from box::Decoder
    float fitness = distances[tour[0] * n + tour[n - 1]];
    for (unsigned i = 1; i < n; ++i)
      fitness += distances[tour[i - 1] * n + tour[i]];
    return fitness;
  }
};
```

### Main
```C++
// main.cpp
#include "TspDecoder.h"
#include <brkga-cuda/Brkga.hpp>
#include <brkga-cuda/BrkgaConfiguration.hpp>
#include <brkga-cuda/DecodeType.hpp>
#include <iostream>
#include <utility>
#include <vector>

int main() {
  // input
  unsigned n;
  std::cin >> n;
  std::vector<std::pair<float, float>> points(n);
  for (unsigned i = 0; i < n; ++i)
    std::cin >> points[i].first >> points[i].second;

  // setup
  TspDecoder decoder(points);
  auto dt = box::DecodeType::fromString("cpu-permutation");
            // or `box::DecodeType(false, false, false)`

  auto config =
      box::BrkgaConfiguration::Builder()
          .decoder(&decoder)
          .decodeType(dt)
          .numberOfPopulations(3)
          .populationSize(256)  // works better with powers of 2
          .chromosomeLength(n)
          .numberOfElites(25)  // or `.elitePercentage(.10)`
          .numberOfMutants(25)  // or `.mutantPercentage(.10)`
          .rhoe(.75)
          .numberOfElitesToExchange(3)
          .seed(1)
          .gpuThreads(256)  // works better with powers of 2
          .ompThreads(6)  // number of CPU threads using OpenMP
          .build();

  box::Brkga brkga(config);
  unsigned maxGenerations = 1000;  // or other stopping condition
  unsigned exchangeInterval = 25;  // optional
  unsigned updateRhoeInterval = 200;  // optional

  // execution
  for (unsigned g = 1; g <= maxGenerations; ++g) {
    brkga.evolve();
    if (g % exchangeInterval == 0)
      brkga.exchangeElites();
    if (g % updateRhoeInterval == 0) {
      // intensifying the search around the elites
      brkga.config.setRhoe(brkga.config.rhoe() + .05);

      // other parameters can be updated too, e.g.:
      //   `brkga.config.setNumberOfElites(10)`
      //   `brkga.config.setOmpThreads(1)`
      //   `brkga.config.setNumberOfElitesToExchange(5)`
    }
  }

  // output
  std::cout << brkga.getBestFitness() << '\n';
  return 0;
}
```
