# Weibull Time To Event prediction with PyTorch and deep learning

This is a PyTorch port and extension of Egil Martinsson's machine learning framework for time to event prediction. In brief, this uses a recurrent neural network to estimate the parameters of the Weibull distribution conditionally modeling the time until a process's next event.

The classes in `wtte/network.py` reimplement the original RNN-based models. I've also experimented with another design based on Transformer architecture, which are implemented in `wtte/transformer.py`. These seem to consistently outperform the original RNN approach.

This also includes sample notebooks and classes that support the Turbofan Degradation Simulation data set that Martinsson and others have benchmarked on. See references for more details.

## Usage

Install as a package using the included `setup.py`, or just use and adapt the Python source as you like!

## Credit and citations

This project is heavily inspired by Egil Martinsson's research and implementation:
https://github.com/ragulpr/wtte-rnn

> Martinsson, Egil. _WTTE-RNN : Weibull Time To Event Recurrent Neural Network._ 2016. Chalmers University Of Technology, Master's Thesis.

It also draws on Dayne Batten's Keras implementation:
https://github.com/daynebatten/keras-wtte-rnn

and Gianmario Spacagna's implementation and discussion:
https://github.com/gm-spacagna/deep-ttf/

Supporting classes for example data are based around:
> A. Saxena and K. Goebel (2008). "Turbofan Engine Degradation Simulation Data Set", https://ti.arc.nasa.gov/c/13/, NASA Ames, Moffett Field, CA.