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

        this.inputWeights = []
        this.reservoirWeights = []
        for (let i = 0; i < reservoirCount; i++) {
            this.inputWeights.push(tf.randomNormal([reservoirSize, inputSize]))

            let reservoirWeights = tf.randomUniform(
                [reservoirSize, reservoirSize],
                -1,
                1
            )
            const norm = reservoirWeights.norm().dataSync()[0]
            reservoirWeights = reservoirWeights.mul(
                tf.scalar(spectralRadius / norm)
            )

            this.reservoirWeights.push(reservoirWeights)
        }

        this.outputWeights = tf.variable(
            tf.randomNormal([outputSize, reservoirSize * reservoirCount])
        )
    }

    predict(input) {
        const batchSize = input.shape[0]
        let reservoirStates = []

        for (let i = 0; i < this.reservoirCount; i++) {
            let reservoir = tf.zeros([batchSize, this.reservoirSize])
            for (let j = 0; j < input.shape[1]; j++) {
                const x = input
                    .slice([0, j], [batchSize, 1])
                    .reshape([batchSize, this.inputSize])
                const prevReservoir = reservoir
                reservoir = tf.tidy(() => {
                    const inputActivation = tf.matMul(
                        x,
                        this.inputWeights[i],
                        false,
                        true
                    )
                    const reservoirActivation = tf.matMul(
                        prevReservoir,
                        this.reservoirWeights[i]
                    )
                    return tf.tanh(inputActivation.add(reservoirActivation))
                })
            }
            reservoirStates.push(reservoir)
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
