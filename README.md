## Avicaching Scripts, Data, Files, Stats -- everything

##### Summer 2017 Project by Anmol Kabra with Yexiang Xue and Prof. Carla Gomes at Cornell University

#### Dependencies:
* Python 2.x (recommend Anaconda)
* NumPy, SciPy
* PyTorch
* `./avicaching_data.py` -- data creator/extractor/mutator/keeper for all models.
* `./lp.py` -- LP implementation for the Pricing Problem's model

#### Identification Problem files:
* `./nnAvicaching_find_weights.py` -- 3-layered neural network
* `./nnAvicaching_find_weights_hiddenlayer.py` -- 4-layered neural network
* For testing:
  - Optimization -- `./runNNAvicaching_weights_orig.sh`
  - GPU SPeedup -- `./runNNAvicaching_weights_rand.sh`

#### Pricing Problem files:
* `./nnAvicaching_find_rewards.py` -- Using algorithm in report
* For testing:
  - Optimization -- `./runNNAvicaching_rewards_orig.sh`
  - GPU Speedup -- `./runNNAvicaching_rewards_rand.sh`
  - Calculate Loss for other baselines -- `./test_rewards.sh`

#### Miscellaneous files:
* `./test_lp_time.py` -- light version of `./nnAvicaching_find_rewards.py` specifically for logging LP runtimes (Appendix B of report)
* `./log_ram_usage_rewards.py` -- logs CPU, RAM, GPU Usage while `./test_lp_time.py` is running
* `./multiple_run.sh` -- runs multiple test scripts on the machine
