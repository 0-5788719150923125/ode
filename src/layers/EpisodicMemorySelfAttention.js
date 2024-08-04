import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

// https://arxiv.org/abs/2407.09450
export default class EpisodicMemorySelfAttention extends LayerBase {
    constructor(config) {
        super(config)
        this.hiddenDim = config.hiddenDim || 256
        this.surpriseThreshold = config.surpriseThreshold || 0.5
        this.memorySize = config.memorySize || 100
        this.knnK = config.knnK || 5
        this.memoryStore = new MemoryStore(this.memorySize)
    }

    build(inputShape) {
        const inputDim = inputShape[inputShape.length - 1]

        this.queryKernel = this.addWeight(
            `queryKernel`,
            [inputDim, this.hiddenDim],
            'float32',
            tf.initializers.glorotUniform()
        )
        this.keyKernel = this.addWeight(
            `keyKernel`,
            [inputDim, this.hiddenDim],
            'float32',
            tf.initializers.glorotUniform()
        )
        this.valueKernel = this.addWeight(
            `valueKernel`,
            [inputDim, inputDim],
            'float32',
            tf.initializers.glorotUniform()
        )
    }

    calculateSurprise(tokens) {
        const surpriseValues = []
        for (let i = 1; i < tokens.shape[1]; i++) {
            const prevTokens = tokens.slice([0, 0], [1, i])
            const currentToken = tokens.slice([0, i], [1, 1])
            const negLogLikelihood = tf.losses
                .sigmoidCrossEntropy(
                    currentToken,
                    prevTokens.matMul(this.valueKernel.read())
                )
                .dataSync()[0]
            surpriseValues.push(negLogLikelihood)
        }

        const mean = tf.mean(surpriseValues).dataSync()[0]
        const std = tf.moments(surpriseValues).variance.sqrt().dataSync()[0]
        const surpriseThreshold = mean + this.surpriseThreshold * std

        const boundaries = []
        for (let i = 0; i < surpriseValues.length; i++) {
            if (surpriseValues[i] > surpriseThreshold) {
                boundaries.push(i)
            }
        }

        const refinedBoundaries = this.refineEventBoundaries(tokens, boundaries)

        const events = []
        for (let i = 0; i < refinedBoundaries.length - 1; i++) {
            const startIndex = refinedBoundaries[i]
            const endIndex = refinedBoundaries[i + 1]
            const eventTokens = tokens.slice(
                [0, startIndex],
                [1, endIndex - startIndex]
            )
            const eventEmbedding = eventTokens.matMul(this.valueKernel.read())
            events.push({ tokens: eventTokens, embedding: eventEmbedding })
        }

        return events
    }

    refineEventBoundaries(tokens, boundaries) {
        const n = tokens.shape[1]
        const m = boundaries.length
        const adjacencyMatrix = tf.zeros([n, n])

        for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
                const similarity = tf.losses
                    .cosineDistance(
                        tokens
                            .slice([0, i], [1, 1])
                            .matMul(this.valueKernel.read()),
                        tokens
                            .slice([0, j], [1, 1])
                            .matMul(this.valueKernel.read())
                    )
                    .dataSync()[0]
                adjacencyMatrix.assign(similarity, [i, j])
                adjacencyMatrix.assign(similarity, [j, i])
            }
        }

        let maxModularity = -Infinity
        let bestBoundaries = boundaries

        for (let i = 0; i < m - 1; i++) {
            for (let j = boundaries[i] + 1; j < boundaries[i + 1]; j++) {
                const newBoundaries = [
                    ...boundaries.slice(0, i),
                    j,
                    ...boundaries.slice(i + 1)
                ]
                const modularity = this.calculateModularity(
                    adjacencyMatrix,
                    newBoundaries
                )

                if (modularity > maxModularity) {
                    maxModularity = modularity
                    bestBoundaries = newBoundaries
                }
            }
        }

        return bestBoundaries
    }

    calculateModularity(adjacencyMatrix, boundaries) {
        const n = adjacencyMatrix.shape[0]
        const m = boundaries.length
        const communities = Array(n).fill(0)

        for (let i = 0; i < m - 1; i++) {
            const startIndex = boundaries[i]
            const endIndex = boundaries[i + 1]
            for (let j = startIndex; j < endIndex; j++) {
                communities[j] = i
            }
        }

        const degrees = tf.sum(adjacencyMatrix, 1).arraySync()
        const totalWeight = adjacencyMatrix.sum().arraySync()

        let modularity = 0

        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                if (communities[i] === communities[j]) {
                    const expected =
                        (degrees[i] * degrees[j]) / (2 * totalWeight)
                    modularity +=
                        adjacencyMatrix
                            .slice([i, j], [1, 1])
                            .arraySync()[0][0] - expected
                }
            }
        }

        return modularity / (2 * totalWeight)
    }

    call(inputs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const events = this.calculateSurprise(inputs)
            events.forEach((event) => this.memoryStore.addEvent(event))

            const retrievedEvents = this.memoryStore.retrieveEvents(
                inputs.matMul(this.valueKernel.read()),
                this.knnK
            )
            const extendedContext = tf.concat(
                [inputs, ...retrievedEvents.map((event) => event.tokens)],
                1
            )

            const Q = this.ops.applyDense(
                extendedContext,
                this.queryKernel.read()
            )
            const K = this.ops.applyDense(
                extendedContext,
                this.keyKernel.read()
            )
            const V = this.ops.applyDense(
                extendedContext,
                this.valueKernel.read()
            )

            const mask = tf.linalg
                .bandPart(
                    tf.ones([
                        extendedContext.shape[1],
                        extendedContext.shape[1]
                    ]),
                    0,
                    -1
                )
                .sub(tf.eye(extendedContext.shape[1]))
                .mul(tf.scalar(-1e9))

            const scores = tf
                .matMul(Q, K, false, true)
                .div(tf.scalar(this.hiddenDim).sqrt())
                .add(mask)

            const weights = scores.softmax()

            let outputs = tf.matMul(weights, V)

            outputs = this.ops.rmsNorm(outputs)

            return tf.add(inputs, outputs)
        })
    }

    getConfig() {
        return {
            ...super.getConfig(),
            hiddenDim: this.hiddenDim,
            surpriseThreshold: this.surpriseThreshold,
            memorySize: this.memorySize,
            knnK: this.knnK
        }
    }
}

class MemoryStore {
    constructor(memorySize) {
        this.memorySize = memorySize
        this.events = []
    }

    addEvent(event) {
        if (this.events.length >= this.memorySize) {
            this.events.shift()
        }
        this.events.push(event)
    }

    retrieveEvents(query, k) {
        const similarities = this.events.map((event) =>
            tf.losses.cosineDistance(query, event.embedding)
        )
        const knnIndices = similarities.topk(k).indices
        const retrievedEvents = knnIndices
            .arraySync()
            .map((index) => this.events[index])

        const contiguityBuffer = []
        retrievedEvents.forEach((event) => {
            const eventIndex = this.events.indexOf(event)
            const prevEvent = this.events[eventIndex - 1]
            const nextEvent = this.events[eventIndex + 1]
            if (prevEvent) contiguityBuffer.push(prevEvent)
            if (nextEvent) contiguityBuffer.push(nextEvent)
        })

        return [...retrievedEvents, ...contiguityBuffer]
    }
}

tf.serialization.registerClass(EpisodicMemorySelfAttention)
