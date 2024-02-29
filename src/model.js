import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-backend-wasm'
import '@tensorflow/tfjs-backend-webgpu'
import '@tensorflow/tfjs-backend-webgl'
import { trainModel } from './train.js'

export default class ModelPrototype {
    constructor(config) {
        this.config = config
        this.vocab = Array.from(
            new Set(
                `¶0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,.?!'"(){}[]|/\\\n `
            )
        )
        this.model = tf.sequential()
    }

    async init() {
        await tf.ready()
        await tf.setBackend(this.config.backend || 'cpu')
        // tf.enableProdMode()
        console.log('Backend:', tf.backend())

        // Add the embedding layer as the first layer
        this.model.add(
            tf.layers.embedding({
                inputDim: this.vocab.length, // Size of the vocabulary
                outputDim: this.config.embeddingDimensions, // Dimension of the embedding vectors
                embeddingsInitializer: 'glorotUniform',
                // embeddingsConstraint: tf.constraints.minMaxNorm({
                //     minValue: -0.02,
                //     maxValue: 0.02
                // })
                trainable: true
            })
        )

        // Add GRU layers
        this.config.layout.forEach((layer, i) => {
            this.model.add(
                tf.layers.bidirectional({
                    layer: tf.layers.gru({
                        units: layer,
                        returnSequences: i < this.config.layout.length - 1, // Set to false for the last GRU layer
                        kernelInitializer: 'glorotUniform',
                        recurrentInitializer: 'orthogonal',
                        biasInitializer: 'zeros',
                        kernelConstraint: tf.constraints.maxNorm({ axis: 0 }),
                        recurrentConstraint: tf.constraints.maxNorm({ axis: 0 })
                    }),
                    mergeMode: 'concat'
                })
            )
        })

        // Add the final Dense layer with softmax activation
        this.model.add(
            tf.layers.dense({
                units: this.vocab.length,
                activation: 'softmax'
            })
        )

        // Compile the model
        this.lossFunction = tf.losses.softmaxCrossEntropy
        this.model.compile({
            optimizer: tf.train.rmsprop(
                this.config.learningRate || 1e-2,
                this.config.decayRate || 0,
                this.config.momentum || 0,
                this.config.epsilon || 1e-8
            ),
            loss: this.lossFunction
        })
    }

    getModel() {
        return this.model
    }

    async train(dataGenerator, batchSize, sampleLen) {
        const bound = trainModel.bind(this)
        await bound(dataGenerator, batchSize, sampleLen)
    }

    getWeights() {
        return this.model.getWeights()
    }

    async generate(seed, temperature = 0.7, length = 20) {
        const bound = generate.bind(this)
        return await bound(seed, temperature, length)
    }

    async saveModel() {
        const savePath = `file://data/model`
        await this.model.save(savePath)
    }
}

async function generate(prompt, temperature = 0.7, maxLength = 20) {
    let generated = prompt
    const tokenIndices = Array.from(prompt).map((e) => this.vocab.indexOf(e))

    for (let i = 0; i < maxLength; i++) {
        const input = tf.tensor2d(
            tokenIndices,
            [1, tokenIndices.length],
            'int32'
        )
        const output = this.model.predict(input).squeeze()

        // Apply temperature scaling by using the log of logits
        const logits = tf
            .log(output)
            .div(tf.scalar(Math.max(temperature, 1e-6)))

        const probabilities = await tf
            .multinomial(logits, 1, null, false)
            .data()

        let winnerIndex = probabilities[0]

        if (winnerIndex < 0 || winnerIndex >= this.vocab.length) {
            winnerIndex = 0 // Fallback to the first index if out of bounds
        }

        const nextChar = this.vocab[winnerIndex]
        generated += nextChar
        tokenIndices.push(winnerIndex)

        tf.dispose([logits, input, probabilities])
    }

    return generated
}
