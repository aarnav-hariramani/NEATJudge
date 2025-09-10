import numpy as np
from neatjudge.evolution.selector import build_features, Selector
import neat

def test_build_features():
    q = np.random.randn(4); q/=np.linalg.norm(q)
    C = np.random.randn(3,4); C/=np.linalg.norm(C,axis=1,keepdims=True)
    X = build_features(q,C)
    assert X.shape==(3,16)

def test_selector_scoring():
    # minimal NEAT config inline
    from neat import config, DefaultGenome, DefaultReproduction, DefaultSpeciesSet, DefaultStagnation
    import tempfile, textwrap
    ini = textwrap.dedent("""[NEAT]
fitness_criterion     = max
fitness_threshold     = 10
pop_size              = 2
reset_on_extinction   = False
[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 5
species_elitism      = 1
[DefaultReproduction]
elitism            = 1
survival_threshold = 0.2
[DefaultGenome]
num_inputs = 16
num_hidden = 4
num_outputs = 1
initial_connection = full_direct
activation_default = tanh
activation_mutate_rate = 0.0
activation_options = tanh
aggregation_default = sum
aggregation_mutate_rate = 0.0
aggregation_options = sum
node_bias_init_mean = 0.0
node_bias_init_stdev = 1.0
bias_init_type = gaussian
bias_mutate_rate = 0.5
bias_mutate_power = 0.5
bias_replace_rate = 0.1
weight_init_mean = 0.0
weight_init_stdev = 1.0
weight_max_value = 30
weight_min_value = -30
weight_mutate_power = 0.5
weight_mutate_rate = 0.8
weight_replace_rate = 0.1
[DefaultSpeciesSet]
compatibility_threshold = 3.0
""")
    import os, tempfile
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write(ini); path=f.name
    cfg = config.Config(DefaultGenome, DefaultReproduction, DefaultSpeciesSet, DefaultStagnation, path)
    g = DefaultGenome(0); g.configure_new(cfg.genome_config)
    s = Selector(g, cfg)
    q = np.random.randn(4); q/=np.linalg.norm(q)
    C = np.random.randn(3,4); C/=np.linalg.norm(C,axis=1,keepdims=True)
    X = build_features(q,C)
    y = s.score(X)
    assert y.shape==(3,)
