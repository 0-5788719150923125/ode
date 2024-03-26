import ModelBase from './model.v0.js'

/**
 * An extremely simple text-generation RNN, used for next-token prediction. You
 * will see versions of this model everywhere, in tutorials on the Internet.
 * @extends ModelBase
 */
export default class ModelPrototype extends ModelBase {
    constructor(config) {
        super(config)
        this.layout = [128, 128, 128]
        this.units = 256
        this.config.mode = 'oneLabel'
    }

    trainTokenizer() {
        super.trainTokenizer(2222, 500_000_000)
    }

    defineBuild() {
        this.model = this.tf.sequential()

        this.model.add(
            this.tf.layers.embedding({
                inputDim: this.tokenizer.getLength(),
                outputDim: this.units,
                maskZero: true
            })
        )

        this.layout.forEach((units, i) => {
            this.model.add(
                this.tf.layers.lstm({
                    units,
                    activation: 'tanh',
                    recurrentActivation: 'sigmoid',
                    returnSequences: i < this.layout.length - 1 // False for the last GRU layer
                })
            )
        })

        this.model.add(
            this.tf.layers.dense({
                units: this.tokenizer.getLength(),
                activation: 'linear'
            })
        )
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
