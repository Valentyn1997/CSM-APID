from hydra.experimental import initialize, compose
import logging

from runnables.train_apid import main

logging.basicConfig(level='info')


class TestAPID:
    def test_apid(self):
        with initialize(config_path="../config"):
            args = compose(config_name="config.yaml", overrides=["+dataset=normal",
                                                                 "+model=apid",
                                                                 "exp.seed=10",
                                                                 "exp.logging=False",
                                                                 "exp.device=cpu",
                                                                 "model.curv_coeff=0.5",
                                                                 "dataset.Y_f=0.0",
                                                                 "model.burn_in_epochs=5",
                                                                 "model.curv_epochs=5",
                                                                 "model.q_epochs=5"])
            results0, results1 = main(args), main(args)
            assert results0 == results1
