import OriginalDecoderEngine from './model.v4.js'

/**
 * A small transformer with multi-head attention and sinusoidal position embeddings.
 * @extends OriginalDecoderEngine
 */
export default class OmniscientDeterministicEnsemble extends OriginalDecoderEngine {
    constructor(config) {
        super(config)
        this.layers = 4
        this.numHeads = 8
        this.units = 256
        this.innerDim = this.units * 4
        this.dropout = 0
        this.epsilon = 1e-5
    }

    build() {
        const inputs = this.tf.input({ shape: [null] })

        const embeddings = this.tf.layers
            .embedding({
                inputDim: this.tokenizer.getLength(),
                outputDim: this.units,
                embeddingsInitializer: 'glorotUniform'
            })
            .apply(inputs)

        const range = this.ode.layers.Range().apply(inputs)

        const encoding = this.ode.layers
            .SinusoidalPositionalEncoding({
                units: this.units,
                reverse: false
            })
            .apply(range)

        let outputs = this.tf.layers.add().apply([embeddings, encoding])

        for (let i = 0; i < this.layers; i++) {
            outputs = this.ode.layers
                .CausalSelfAttention({
                    blockSize: this.config.contextLength,
                    units: this.units,
                    numHeads: this.numHeads,
                    dropout: this.dropout,
                    bias: false,
                    epsilon: this.epsilon
                })
                .apply(outputs)

            // outputs = this.ode.layers
            //     .SynthesizerAttention({
            //         blockSize: this.config.contextLength,
            //         units: this.units,
            //         numHeads: this.numHeads,
            //         dropout: this.dropout,
            //         bias: false,
            //         epsilon: this.epsilon
            //     })
            //     .apply(outputs)

            // outputs = this.ode.layers
            //     .GaussianMixtureModel({
            //         units: this.units,
            //         innerDim: this.innerDim,
            //         numHeads: this.numHeads,
            //         dropout: this.dropout,
            //         epsilon: this.epsilon,
            //         activation: 'swish'
            //     })
            //     .apply(outputs)

            outputs = this.ode.layers
                .MultiLayerPerceptron({
                    units: this.units,
                    innerDim: this.innerDim,
                    numHeads: this.numHeads,
                    dropout: this.dropout,
                    epsilon: this.epsilon,
                    activation: 'swish'
                })
                .apply(outputs)
        }

        outputs = this.tf.layers
            .layerNormalization({
                epsilon: this.epsilon
            })
            .apply(outputs)

        outputs = this.tf.layers
            .dense({
                units: this.tokenizer.getLength(),
                activation: 'linear'
            })
            .apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }

    async generate(prompt, temperature = 0.7, length = 20) {
        return await generateText.call(this, prompt, temperature, length)
    }
}

async function generateText(prompt, temperature, maxNewTokens) {
    let inputs = await prepareInputs.call(this, this.tokenizer.encode(prompt))
    // Adjust this part of your generateText function
    for (let step = 0; step < maxNewTokens; step++) {
        const idxNext = generateOnce.call(this, inputs, temperature)
        // Ensure idxNext has a shape of [1, 1] to match the rank of inputs
        const idxNextExpanded = idxNext.expandDims(1) // Adjusting idxNext shape for concatenation
        const idxNew = this.tf.concat([inputs, idxNextExpanded], 1) // Adjusting the axis to 1 for correct concatenation
        this.tf.dispose(inputs)
        inputs = idxNew
        this.tf.dispose(idxNext)
    }

    const idxArr = await inputs.array()
    this.tf.dispose(inputs)
    return this.tokenizer.decode(idxArr[0])
}

function generateOnce(idx, temperature) {
    let idxNext
    this.tf.tidy(() => {
        const block_size = this.model.inputs[0].shape[1]
        const idxCond =
            idx.shape[1] <= block_size
                ? idx
                : idx.slice([0, -block_size], [-1, -1])
        // Forward the model to get the logits for the index in the sequence
        const logits = this.model.predict(idxCond)

        let logitsScaled
        if (logits.shape.length === 3) {
            // Assuming timeDistributed mode if logits shape is 3D
            // pluck the logits at the final step for timeDistributed
            logitsScaled = logits
                .slice([0, idx.shape[1] - 1, 0], [1, 1, logits.shape[2]])
                .reshape([logits.shape[2]])
        } else {
            // singleLabel mode
            // For singleLabel mode, logits is already in the expected shape
            logitsScaled = logits
        }

        // either sample from the distribution or take the most likely element
        if (temperature !== 1) {
            // scale by desired temperature
            logitsScaled = logitsScaled.div(this.tf.scalar(temperature))
            idxNext = this.tf.multinomial(logitsScaled, 1).reshape([-1])
        } else {
            idxNext = logitsScaled.argMax(-1).expandDims(-1)
        }

        this.tf.keep(idxNext)
    })
    return idxNext
}

function prepareInputs(inputs) {
    this.tf.tidy(() => {
        // Check if idx is a tensor or an array
        if (inputs instanceof this.tf.Tensor) {
            inputs = inputs.clone()
        } else {
            inputs = this.tf.tensor(inputs)
        }
        // Check data type
        if (inputs.dtype !== 'int32') {
            inputs = inputs.toInt()
        }
        // If the shape of idx is 1D, we need to add a dimension
        if (inputs.shape.length === 1) {
            inputs = inputs.expandDims(0)
        }
        this.tf.keep(inputs)
        // keep idx from deletion
    })
    return inputs
}
