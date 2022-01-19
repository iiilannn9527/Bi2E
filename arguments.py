ARGUMENT_1 = {"learn rate": 0.00025, "max_steps": 120000, "warm_up_steps": 30000, "valid_steps": 10000, "log_steps": 10000,
              "damping_steps": 20000, "damping_rate": 2.0,
              "negative_adversarial_sampling": True, "negative sample size": 512,
              "train batch size": 512, "test batch size": 32,
              "hidden dim": 500, "gamma": 2.5,
              "adversarial_temperature": 1.5, }

ARGUMENT_K = {"learn rate": 0.00025, "max_steps": 120000, "warm_up_steps": 30000, "valid_steps": 10000, "log_steps": 10000,
              "negative_adversarial_sampling": True, "negative sample size": 382,
              "train batch size": 1024, "test batch size": 16,
              "hidden dim": 1500, "gamma": 7.5,
              "adversarial_temperature": 1.25}

ARGUMENT_Y = {"learn rate": 0.00025, "max_steps": 150000, "warm_up_steps": 30000, "valid_steps": 10000, "log_steps": 10000,
              "damping_steps": 30000, "damping_rate": 2.0,
              "negative_adversarial_sampling": True, "negative sample size": 400,
              "train batch size": 1024, "test batch size": 4,
              "hidden dim": 500, "gamma": 24.0,
              "adversarial_temperature": 0.5}

ARGUMENT = {"arg_1": ARGUMENT_1,
            'arg_k': ARGUMENT_K,
            'arg_y': ARGUMENT_Y
            }
