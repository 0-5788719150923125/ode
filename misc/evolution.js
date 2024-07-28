import * as tf from '@tensorflow/tfjs'

class SimpleNeuralNetwork {
    constructor(inputSize, hiddenSize, outputSize) {
        this.model = tf.sequential()
        this.model.add(
            tf.layers.dense({
                units: hiddenSize,
                inputShape: [inputSize],
                activation: 'tanh'
            })
        )
        this.model.add(
            tf.layers.dense({ units: outputSize, activation: 'tanh' })
        )
    }

    predict(X) {
        return this.model.predict(X)
    }

    getWeights() {
        return this.model.getWeights()
    }

    setWeights(weights) {
        this.model.setWeights(weights)
    }
}

function mutate(weights, mutationRate = 0.1, mutationScale = 0.1) {
    return weights.map((w) => {
        const shape = w.shape
        const flatW = w.flatten()
        const mutation = tf.randomNormal(flatW.shape).mul(mutationScale)
        const mutationMask = tf.randomUniform(flatW.shape).less(mutationRate)
        const mutatedFlat = flatW.add(mutation.mul(mutationMask))
        return mutatedFlat.reshape(shape)
    })
}

function crossover(parent1Weights, parent2Weights) {
    return parent1Weights.map((w1, i) => {
        const w2 = parent2Weights[i]
        const shape = w1.shape
        const flat1 = w1.flatten()
        const flat2 = w2.flatten()
        const crossoverPoint = Math.floor(Math.random() * flat1.size)
        const firstHalf = flat1.slice([0], [crossoverPoint])
        const secondHalf = flat2.slice([crossoverPoint])
        return tf.concat([firstHalf, secondHalf]).reshape(shape)
    })
}

function evaluateFitness(network, X, y) {
    const predictions = network.predict(X)
    const mse = tf.losses.meanSquaredError(y, predictions)
    return tf.scalar(1).div(mse.add(1))
}

async function evolutionaryTraining(populationSize, generations, X, y) {
    const inputSize = X.shape[1]
    const outputSize = y.shape[1]
    const hiddenSize = 5

    let population = Array(populationSize)
        .fill()
        .map(() => new SimpleNeuralNetwork(inputSize, hiddenSize, outputSize))

    for (let generation = 0; generation < generations; generation++) {
        // Evaluate fitness
        const fitnesses = await Promise.all(
            population.map((network) => evaluateFitness(network, X, y).array())
        )

        // Select parents
        const parents = tf.util
            .createShuffledIndices(populationSize)
            .sort((a, b) => fitnesses[b] - fitnesses[a])
            .slice(0, populationSize / 2)

        // Create next generation
        const newPopulation = []
        for (let i = 0; i < populationSize; i += 2) {
            const parent1 = population[parents[i % parents.length]]
            const parent2 = population[parents[(i + 1) % parents.length]]

            const child1Weights = mutate(
                crossover(parent1.getWeights(), parent2.getWeights())
            )
            const child2Weights = mutate(
                crossover(parent2.getWeights(), parent1.getWeights())
            )

            const child1 = new SimpleNeuralNetwork(
                inputSize,
                hiddenSize,
                outputSize
            )
            const child2 = new SimpleNeuralNetwork(
                inputSize,
                hiddenSize,
                outputSize
            )

            child1.setWeights(child1Weights)
            child2.setWeights(child2Weights)

            newPopulation.push(child1, child2)
        }

        population = newPopulation

        if (generation % 10 === 0) {
            const bestFitness = Math.max(...fitnesses)
            console.log(
                `Generation ${generation}: Best Fitness = ${bestFitness}`
            )
        }
    }

    const finalFitnesses = await Promise.all(
        population.map((network) => evaluateFitness(network, X, y).array())
    )
    return population[finalFitnesses.indexOf(Math.max(...finalFitnesses))]
}

// Example usage
async function run() {
    const X = tf.randomNormal([100, 2])
    const y = X.slice([0, 0], [-1, 1])
        .greater(X.slice([0, 1], [-1, 1]))
        .cast('float32')

    const population = 50
    const generations = 10000
    const bestNetwork = await evolutionaryTraining(
        population,
        generations,
        X,
        y
    )
    console.log('Training complete. Best network found.')

    // Test the best network
    const testX = tf.randomNormal([10, 2])
    const predictions = bestNetwork.predict(testX)
    console.log('Test predictions:', await predictions.array())
}

run()
