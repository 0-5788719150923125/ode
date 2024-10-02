# ode

omniscient deterministic engine

# features

This library implements:

-   a suite of experimental language models, including transformers, recurrent neural networks, state space models, and more
-   a plethora of custom, composable layer types, for use in your own models
-   custom optimizers, including AdamW, Lion, and Prodigy
-   learning rate schedulers
-   custom loss functions
-   a custom training loop, with gradient accumulation, gradient checkpointing (broken), L2-normalized gradient clipping
-   custom tokenizers, including TokenMonster and support for all Huggingface Tokenizers models
-   dataset management via generator functions and iterable sampling strategies
-   a number of text-decoding strategies, including greedy (argmax), temperature, top-k, top-p, and Mirostat sampling
-   metrics logging and visualization
-   object-oriented, extensible design - with functional architecture and operations composition
-   tons more

# usage

See [cli.js](./cli.js) for complete usage.

## CLI Example:

```sh
node cli.js \
  --version 6 \
  --batchSize 2 \
  --gradientAccumulationSteps 8 \
  --sampleLength 256 \
  --generateEvery 512 \
  --predictLength 512 \
  --saveEvery 250 \
  --action train
```

## Library Example:

```js
import ODE from 'ode'

const net = await ODE({
    backend: 'webgl', // available backends: ['cpu', 'tensorflow', 'webgl', 'webgpu']
    version: 6
})

await net.init()

const dataSampler = net.ode.samplers.CosmopediaSampler()
await net.train(dataSampler, {
    batchSize: 1,
    gradientAccumulationSteps: 64,
    sampleLength: 256,
    saveEvery: 100
})

const output = await net.generate({
    prompt: 'Once upon a time, ',
    doSample: true,
    temperature: 0.7,
    maxNewTokens: 64,
    repetitionPenalty: 1.2
})

console.log(output)
```

## Run tests:

```sh
npm run test
npm run test:suite --suite=models
```

## Produce Metrics:

```sh
python metrics_visualizer.py --label selfModel auxiliaryWeight
```
