ARGUMENT_0 = {"learn rate": 0.0005, "max_steps": 80000, "warm_up_steps": 10000, "valid_steps": 10000, "log_steps": 10000,
              "negative_adversarial_sampling": True, "negative sample size": 512,
              "train batch size": 512, "test batch size": 12,
              "hidden dim": 1000, "gamma": 6.0,
              "adversarial_temperature": 0.5}

ARGUMENT_1 = {"learn rate": 0.00025, "max_steps": 120000, "warm_up_steps": 30000, "valid_steps": 10000, "log_steps": 10000,
              "damping_steps": 20000, "damping_rate": 2.0,
              "negative_adversarial_sampling": True, "negative sample size": 512,
              "train batch size": 512, "test batch size": 32,
              "hidden dim": 500, "gamma": 2.5,
              "adversarial_temperature": 1.5, }

ARGUMENT_2 = {"learn rate": 0.0005, "max_steps": 60000, "warm_up_steps": 30000, "valid_steps": 3000,
              "log_steps": 100000,
              "negative_adversarial_sampling": True, "negative sample size": 512,
              "train batch size": 512, "test batch size": 64,
              "hidden dim": 500, "gamma": 9.0,
              "adversarial_temperature": 1.}

# bash run.sh train RotatE wn18rr 0 0 512 1024 500 6.0 0.5 0.00005 80000 8 -de
ARGUMENT_R = {"learn rate": 0.00005, "max_steps": 80000, "warm_up_steps": 160000, "valid_steps": 20000, "log_steps": 10000,
              "negative_adversarial_sampling": True, "negative sample size": 1024,
              "train batch size": 512, "test batch size": 16,
              "hidden dim": 500, "gamma": 6.0,
              "adversarial_temperature": 0.5}

# bash run.sh train RotatE FB15k-237 0 0 1024 256 1000 9.0 1.0 0.00005 100000 16 -de
# bash run.sh train PairRE FB15k-237 0 0 1024 256 1500 6.0 1.0 0.00005 100000 16 -dr
ARGUMENT_K = {"learn rate": 0.00025, "max_steps": 120000, "warm_up_steps": 30000, "valid_steps": 10000, "log_steps": 10000,
              "negative_adversarial_sampling": True, "negative sample size": 382,
              "train batch size": 1024, "test batch size": 16,
              "hidden dim": 1500, "gamma": 7.5,
              "adversarial_temperature": 1.25}

# bash run.sh train TransE wn18rr 0 0 512 1024 500 6.0 0.5 0.00005 80000 8
ARGUMENT_T = {"learn rate": 0.00005, "max_steps": 80000, "warm_up_steps": 160000, "valid_steps": 20000, "log_steps": 10000,
              "negative_adversarial_sampling": True, "negative sample size": 1024,
              "train batch size": 512, "test batch size": 64,
              "hidden dim": 500, "gamma": 6.0,
              "adversarial_temperature": 0.5}

# bash run.sh train pRotatE wn18rr 0 0 512 1024 500 6.0 0.5 0.00005 80000 8
ARGUMENT_P = {"learn rate": 0.00005, "max_steps": 80000, "warm_up_steps": 40000, "valid_steps": 20000, "log_steps": 10000,
              "negative_adversarial_sampling": True, "negative sample size": 1024,
              "train batch size": 512, "test batch size": 32,
              "hidden dim": 500, "gamma": 6.0,
              "adversarial_temperature": 0.5}

# bash run.sh train RotatE YAGO3-10 0 0 1024 400 500 24.0 1.0 0.0002 100000 4 -de
ARGUMENT_Y = {"learn rate": 0.00025, "max_steps": 150000, "warm_up_steps": 30000, "valid_steps": 10000, "log_steps": 10000,
              "damping_steps": 30000, "damping_rate": 2.0,
              "negative_adversarial_sampling": True, "negative sample size": 400,
              "train batch size": 1024, "test batch size": 4,
              "hidden dim": 500, "gamma": 24.0,
              "adversarial_temperature": 0.5}

ARGUMENT = {"arg_0": ARGUMENT_0,
            "arg_1": ARGUMENT_1,
            "arg_2": ARGUMENT_2,
            "arg_r": ARGUMENT_R,
            'arg_k': ARGUMENT_K,
            'arg_t': ARGUMENT_T,
            'arg_p': ARGUMENT_P,
            'arg_y': ARGUMENT_Y
            }
