import * as tf from '@tensorflow/tfjs-node'

class EchoStateNetwork {
    constructor(
        inputSize,
        reservoirSize,
        outputSize,
        reservoirCount,
        spectralRadius
    ) {
        this.inputSize = inputSize
        this.reservoirSize = reservoirSize
        this.outputSize = outputSize
        this.reservoirCount = reservoirCount

        this.reservoirModels = []
        for (let i = 0; i < reservoirCount; i++) {
            this.reservoirModels.push(
                createReservoirModel(inputSize, reservoirSize, spectralRadius)
            )
        }

        this.outputWeights = tf.variable(
            tf.randomNormal([outputSize, reservoirSize * reservoirCount])
        )
    }

    predict(input) {
        const batchSize = input.shape[0]
        let reservoirStates = []

        for (let i = 0; i < this.reservoirCount; i++) {
            const reservoirOutput = this.reservoirModels[i].predict(input)
            reservoirStates.push(reservoirOutput)
        }

        const combinedStates = tf.concat(reservoirStates, 1)
        return tf.matMul(combinedStates, this.outputWeights.transpose())
    }

    async train(input, target, learningRate, epochs) {
        const optimizer = tf.train.adam(learningRate)

        for (let i = 0; i < epochs; i++) {
            const cost = optimizer.minimize(
                () => {
                    const output = this.predict(input)
                    const loss = tf.losses.meanSquaredError(target, output)
                    return loss
                },
                true,
                [this.outputWeights]
            )

            await cost.data()
            cost.dispose()
            await tf.nextFrame()
        }
    }
}

class ReservoirCell extends tf.layers.Layer {
    constructor(units, spectralRadius, activation = 'tanh') {
        super({ name: 'esn' })
        this.units = units
        this.spectralRadius = spectralRadius
        this.activation = activation
    }

    build(inputShape) {
        const inputDim = inputShape[inputShape.length - 1]
        this.kernel = this.addWeight(
            'kernel',
            [inputDim, this.units],
            'float32',
            tf.initializers.randomUniform({ minval: -1, maxval: 1 }),
            null,
            true
        )
        this.recurrentKernel = this.addWeight(
            'recurrent_kernel',
            [this.units, this.units],
            'float32',
            tf.initializers.randomUniform({ minval: -1, maxval: 1 }),
            null,
            true
        )
        this.built = true
    }

    call(inputs, states) {
        inputs = Array.isArray(inputs) ? inputs[0] : inputs
        let h = states[0] || tf.zeros([inputs.shape[0], this.units])
        const output = tf.tidy(() => {
            const newState = tf.tanh(
                tf.add(
                    tf.matMul(inputs, this.kernel.read()),
                    tf.matMul(h, this.recurrentKernel.read())
                )
            )
            return [newState, newState]
        })
        return output
    }

    getConfig() {
        const config = super.getConfig()
        Object.assign(config, {
            units: this.units,
            activation: this.activation
        })
        return config
    }

    get stateSize() {
        return this.units
    }
}

function createReservoirModel(inputSize, reservoirSize, spectralRadius) {
    const input = tf.input({ shape: [null, inputSize] })
    const reservoir = tf.layers
        .rnn({
            cell: new ReservoirCell(reservoirSize, spectralRadius),
            returnSequences: false
        })
        .apply(input)
    const model = tf.model({ inputs: input, outputs: reservoir })

    const recurrentKernel = model.getWeights()[1]
    const norm = tf.norm(recurrentKernel, 'euclidean')
    const scaledKernel = recurrentKernel
        .div(norm)
        .mul(tf.scalar(spectralRadius))
    model.setWeights([model.getWeights()[0], scaledKernel])

    return model
}

// Example usage
const inputSize = 1
const reservoirSize = 100
const outputSize = 1
const reservoirCount = 5
const spectralRadius = 0.9

const esn = new EchoStateNetwork(
    inputSize,
    reservoirSize,
    outputSize,
    reservoirCount,
    spectralRadius
)

// Generate random training data
const batchSize = 10
const sequenceLength = 20
const input = tf.randomNormal([batchSize, sequenceLength, inputSize])
const target = tf.randomNormal([batchSize, outputSize])

// Train the ESN
await esn.train(input, target, 0.01, 100)

// Make predictions
const testInput = tf.randomNormal([1, sequenceLength, inputSize])
const output = esn.predict(testInput)
console.log(output.dataSync())
