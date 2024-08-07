# ode

omniscient deterministic engine

# usage

See: [cli.js](./cli.js)

# serial experiments

The Second Law of Thermodynamics states that the energy of a system always decreases over time, and that the entropy of a system always increases... this translates to... a `${temperature}` increase at every sequential layer. Lower layers should become deterministic and stable, while later layers are chaotic and unpredictable.

A Mixture of Optimizers (MoO) is...

# features

This library implements:

-   a suite of experimental language models, including transformers, recurrent neural networks, state space models, and more
-   a plethora of custom, composable layer types, for use in your own models
-   custom optimizers, including AdamW, Lion, and Prodigy
-   learning rate schedulers
-   custom loss functions
-   a custom training loop, with gradient checkpointing (broken), L2-normalized gradient clipping
-   custom tokenizers, including support for all Huggingface Tokenizers models
-   dataset management via generator functions and iterable sampling strategies
-   support for one-label, multi-label, one-hot and integer-encoded datasets
-   a number of text-decoding strategies, including greedy (argmax), temperature, top-k and top-p sampling
-   object-oriented, extensible design
-   tons more

# todo

-   [] in cross attention, queries are generated from one embedding, and keys/values from another (A "peer" is all you need)
-   [] explore model parallelism, sharding, peer-to-peer capabilities
-   [] reinforcement learning, to teach the models "when" to respond, and when to remain quiet
-   [] cross-modal models (i.e. a picture of text predicts a character)
