# GFlowNets - Automatic Redistricting to Solve Gerrymandering

## Sections
- [GFlowNets - Automatic Redistricting to Solve Gerrymandering](#gflownets---automatic-redistricting-to-solve-gerrymandering)
  - [Sections](#sections)
  - [Introduction](#introduction)
  - [Problem](#problem)
  - [Data](#data)
  - [Methodology](#methodology)
  - [Results](#results)
  - [Conclusion](#conclusion)
  - [References](#references)
  - [Contributors](#contributors)
  - [License](#license)

## Introduction

Gerrymandering is the practice of manipulating the boundaries of an electoral constituency to favor one party or class. This project aims to reduce the impact of gerrymandering by creating a tool that can be used to draw fair and unbiased electoral districts. The tool will use Generative Flow Networks (GFNs) to generate a set of possible district boundaries that are both contiguous and compact. The tool will also use a set of fairness metrics to evaluate the generated districts and select the most fair and unbiased districting plan.

## Problem

The problem of gerrymandering is a significant issue in the United States. The goal of this project is to create a tool that can be used to draw fair and unbiased electoral districts that represent the interests of all voters.

## Data

We use the following website to find the data for the congressional districts of any state in the US:
https://alarm-redist.org/fifty-states/PA_cd_2020/

We use the data or Pennsylvania like in the reference paper.

The data has been extracted and saved in json format using jsonlite in R.

## Methodology

MCMC Sampling + Generative Flow Networks

## Results

## Conclusion

## References
- [Automated Redistricting Simulation Using Markov Chain Monte Carlo](https://imai.fas.harvard.edu/research/files/redist.pdf)

## Contributors
- [Alex Maggioni](alex.maggioni@mila.quebec)
- [Arnaud Bergeron](arnaud.bergeron@mila.quebec)
- [Kevin Lessard](kevin.lessard@mila.quebec)

## License