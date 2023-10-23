PartialCounterfactualIdent
==============================

Partial counterfactual identification for continuous outcomes with a Curvature Sensitivity Model

The project is built with following Python libraries:
1. [Pyro](https://pyro.ai/) - deep learning and probabilistic modeling with residual normalizing flows and variational augmentations
2. [Hydra](https://hydra.cc/docs/intro/) - simplified command line arguments management
3. [MlFlow](https://mlflow.org/) - experiments tracking
4. [POT](https://pythonot.github.io/) - pytorch implementation of the Wasserstein distance
5. [normflows](https://pypi.org/project/normflows/) - pytorch implementation of the residual normalizing flows

### Installations
First one needs to make the virtual environment and install all the requirements:
```console
pip3 install virtualenv
python3 -m virtualenv -p python3 --always-copy venv
source venv/bin/activate
pip3 install -r requirements.txt
```

## MlFlow Setup / Connection
To start an experiments server, run: 

`mlflow server --port=5000`

To access MlFLow web UI with all the experiments, connect via ssh:

`ssh -N -f -L localhost:5000:localhost:5000 <username>@<server-link>`

Then, one can go to local browser http://localhost:5000.

## Real-world case study

We use multi-country data from [Banholzer et al. (2021)](https://doi.org/10.1371/journal.pone.0252827). Access data [here](https://github.com/nbanho/npi_effectiveness_first_wave/blob/master/data/data_preprocessed.csv).

## Experiments

Main training script of **Augmented Pseudo-Invertible Decoder** (APID) is universal for different datasets. For details on mandatory arguments - see the main configuration file `config/config.yaml` and other files in `configs/` folder.

Generic script with logging and fixed random seed is following:
```console
PYTHONPATH=.  python3 runnables/train_apid.py +dataset=<dataset> +model=apid exp.seed=10 exp.logging=True
```

Example of running APID on multi-modal dataset with curvature loss coefficient $\lambda_\kappa = 5.0$ and factual outcomes $y' = 0.0,0.5,1.0,1.5,2.0$:
```console
PYTHONPATH=.  python3 runnables/train_apid.py +dataset=multi_modal +model=apid exp.seed=10 exp.logging=True model.curv_coeff=5.0 dataset.Y_f=0.0,0.5,1.0,1.5,2.0
```



--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
