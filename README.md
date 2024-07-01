# **TraPy**
### *By Nikhil Lalgudi Vaidyanathan*

An all-in-one library for any form of technical analysis/advanced statistical methods for the any form of market trading. The original goal is to create an all-in-one library so that I can teach my mom how to use pandas and numpys to trade. This repository is open source and contains most well-known technical indicators for technical analysis of the market.

## **Core Features**

### **Indicators**
- **Graph-trends**: Bollinger Bands, Donchian Channels, fractal chaos bands, keltner channels, moving average envelopes, starc bands, price channels, pivot points, rolling pivot points, standard deviation channels, Ichimoku Cloud, Parabolic SAR
- **Moving averages**: alma, dema, epma, ema, hilbert transform instantaneous trendline, hull moving average, Kama, lsma, mesa adaptive moving average, mcginley dynamic, modified moving average, running moving average, simple moving average, smoothed moving average, tillson t3 moving average, tema, vwap, vwma, wma, frama, zlema, tema
- **Oscillators**: awesome oscillator, cmo, cci, Connors rsi, dpo, kdj_index, RSI, schaff_trend_cycle, stochastic_momentum_index, stochastic_oscillator, stochastic_rsi, trix (triple_ema_oscillator), ultimate_oscillator, williams_r, Elder-Ray Index, RVI
- **Price Characteristics**: Average True Range, Balance of Power, Bull-Bear Power, Choppiness Index, Dominant Cycle Periods, Historical Volatility, Hurst Exponent, Momentum Oscillator, Normalized Average True Range, Price Momentum Oscillator, Price Relative Strength, Rate of Change, Rescaled Range Analysis, True Range, True Strength Index, Ulcer Index, Moving Average Crossover (and its plotting function), Commodity Channel Index, Directional Movement Index, Vortex Indicator, Beta Coefficient, Correlation Coefficient, Linear Regression, Mean Absolute Deviation, Mean Absolute Percentage Error, Mean Square Error, R-squared, Standard Deviation, Z-score
- **Transforms**: ehlers_fisher_transform, heikin_ashi, renko, zigzag_indicator. Kagi Charts, Line Break Charts

### **Stochastic Realizations**
#### **Continuous Processes**
- Bessel
- Brownian Bridge
- Brownian Excursion
- Brownian Meander
- Brownian Motion
- Cauchy
- Fractional Brownian Motion
- Gamma
- Geometric Brownian Motion
- Hyperbolic Brownian Motion
- Inverse Gaussian 
- Lévy Flight
- Lévy Process
- Martingale
- Mixed Poisson
- Multifractional Brownian Motion
- Normal Inverse Gaussian
- Poisson
- Stable Processes
- Squared Bessel
- Variance Gamma
- Wiener

#### **Diffusion Processes**
- Constant Elasticity Variance
- Cox Ingersoll Ross
- Diffusion
- Extended Vasicek
- Multifactor Vasicek
- Vasicek
- Ornstein Ulhenbeck
- Mean-Reverting 

#### **Discrete Processes**
- Bernoulli 
- Chinese Restaurant
- Dirichlet 
- Galton-Watson
- Markov
- Moran
- Markov Chain Monte Carlo
- Random Walk
- Renewal Process
- Yule Process

#### **Noise**
- Alpha-Stable Noise
- Colored Noise
- Fractional Gaussian
- Gaussian Noise

### **Numerical Financial Methods**
- Black Scholes Numerical Methods
- Monte Carlo Methods
- Quasi Monte Carlo Methods
- FFT Methods
- Mean Variance Optimization
- Binomial model
- Finite Difference Methods
- Pseudo-Random Numbers
- Kalman Filter
- Trinomial Methods
- Non-linear Solvers
- Linear Programming
- Unconstrained Optimization

## **Interactive Indicators**

## **Advanced Statistical Methods**
### **Solvers of different differential equations with GPU support and efficient backpropagation with PyTorch**

#### **Delay Differential Equations:**
- Differential algebraic equations (DAEs)

#### **Differential algebraic equations:**
- Delay differential equations (DDEs)

#### **Discrete Equations:**
- Discrete equations (function maps, discrete stochastic (Gillespie/Markov) simulations)

#### **Mixed discrete and continuous equations:**
- Mixed discrete and continuous equations (Hybrid Equations, Jump Diffusions)

#### **Neutral, retarded, and algebraic delay differential equations:**
- Neutral, retarded, and algebraic delay differential equations (NDDEs, RDDEs, and DDAEs)

#### **Ordinary Differential Equations:**
- Ordinary differential equations (ODEs)
- Split and Partitioned ODEs (Symplectic integrators, IMEX Methods)

#### **Random Differential Equations:**
- Random differential equations (RODEs or RDEs)

#### **Stochastic Differential Equations:**
- (Stochastic) partial differential equations ((S)PDEs) (with both finite difference and finite element methods)
- Stochastic delay differential equations (SDDEs)
- Stochastic ordinary differential equations (SODEs or SDEs)
- Stochastic differential-algebraic equations (SDAEs)
- Experimental support for stochastic neutral, retarded, and algebraic delay differential equations (SNDDEs, SRDDEs, and SDDAEs)

#### **Integro-Differential Equations:**
Include solvers for equations involving both integrals and derivatives.

#### **Fractional Differential Equations:**
Support for equations involving fractional derivatives, which are useful in various fields such as viscoelasticity and anomalous diffusion.

#### **Hamiltonian Systems**
Specialized solvers for systems governed by Hamiltonian mechanics.

#### **Reaction-Diffusion Systems:**
Efficient solvers for modeling chemical reactions and biological processes.

## **Getting Started**
### **Prerequisites**
Make sure you have at least Python 3.6 installed, along with the following packages:
- **numpy**
- **pandas**
- **matplotlib**

You can install the required packages using the following command:
```bash
pip install numpy pandas matplotlib
```

### **Installation**
To get started, clone the repository using the following command:
```bash
git clone https://github.com/nikhil-lalgudi/TraPy.git
```

## **Contributing**
We welcome contributions from the community! If you would like to contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a pull request.

## **Feedback and Ideas**
We'd love to hear from you! If you have any feedback, new ideas, or just want to discuss the project, feel free to reach out to us at [nikhillv@umich.edu](mailto:nikhillv@umich.edu).

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

