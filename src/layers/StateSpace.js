import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

// This is not a very good state space model. This one is recurrent and
// slow, while modern designs are fast and parallelizable.
export default class StateSpace extends LayerBase {
    constructor(config) {
        super({ name: `ssm-${randomString()}`, ...config })
        this.units = config.units || 64
        this.innerDim = config.innerDim || 256
        this.returnSequences = config.returnSequences || false
        this.decayFactor = config.decayFactor || 1.0
        this.activation = config.activation || 'tanh'
        this.beta = config.beta || 1.0
    }

    build(inputShape) {
        const inputDim = inputShape[2]
        this.kernel = this.addWeight(
            'kernel',
            [inputDim, this.innerDim],
            'float32',
            tf.initializers.glorotNormal()
        )
        this.recurrentKernel = this.addWeight(
            'recurrentKernel',
            [this.units, this.innerDim],
            'float32',
            tf.initializers.orthogonal({ gain: 1 })
        )
        this.outputKernel = this.addWeight(
            'outputKernel',
            [this.units, this.units],
            'float32',
            tf.initializers.glorotNormal()
        )
        this.bias = this.addWeight(
            'bias',
            [this.innerDim],
            'float32',
            tf.initializers.zeros()
        )
        this.meanKernel = this.addWeight(
            'meanKernel',
            [this.innerDim, this.units],
            'float32',
            tf.initializers.glorotNormal()
        )
        this.meanBias = this.addWeight(
            'meanBias',
            [this.units],
            'float32',
            tf.initializers.zeros()
        )
        this.logVarKernel = this.addWeight(
            'logVarKernel',
            [this.innerDim, this.units],
            'float32',
            tf.initializers.glorotNormal()
        )
        this.logVarBias = this.addWeight(
            'logVarBias',
            [this.units],
            'float32',
            tf.initializers.zeros()
        )
    }

    sampleLatentState(innerState, kwargs) {
        return tf.tidy(() => {
            const mean = tf.add(
                tf.matMul(innerState, this.meanKernel.read()),
                this.meanBias.read()
            )
            const logVar = tf.add(
                tf.matMul(innerState, this.logVarKernel.read()),
                this.logVarBias.read()
            )
            const expLogVar = logVar.exp()

            if (kwargs.training) {
                // Compute the KL Divergence
                const klDivergence = logVar
                    .add(1)
                    .sub(mean.square())
                    .sub(expLogVar)
                    .mean()
                    .mul(-0.5)
                    .mul(this.beta)

                // Add it to the loss function
                if (!this.extraLoss) this.extraLoss = tf.keep(klDivergence)
                else {
                    const oldValue = this.extraLoss
                    this.extraLoss = tf.keep(this.extraLoss.add(klDivergence))
                    oldValue.dispose()
                }
            }

            // Sample from the latent space using the reparameterization trick
            const epsilon = tf.randomNormal(mean.shape)
            return tf.add(mean, tf.mul(epsilon, expLogVar.sqrt()))
        })
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const [batchSize, sequenceLength, inputDim] = inputs.shape

            let state = tf.zeros([batchSize, this.units])
            const outputs = []

            const kernel = this.kernel.read()
            const recurrentKernel = this.recurrentKernel.read()
            const outputKernel = this.outputKernel.read()
            const bias = this.bias.read()

            for (let t = 0; t < sequenceLength; t++) {
                const input = inputs
                    .slice([0, t, 0], [batchSize, 1, inputDim])
                    .reshape([batchSize, inputDim])
                const innerState = tf
                    .add(
                        tf.matMul(input, kernel),
                        tf.matMul(state, recurrentKernel).mul(this.decayFactor)
                    )
                    .add(bias)
                const activatedState = tf.layers
                    .activation({ activation: this.activation })
                    .apply(innerState)
                const latentState = tf.tidy(() => {
                    return this.sampleLatentState(activatedState, kwargs)
                })
                const newState = tf.matMul(latentState, outputKernel)
                outputs.push(newState)
                state = newState
            }

            let output = this.returnSequences
                ? tf.stack(outputs, 1)
                : outputs[outputs.length - 1]

            output = this.ops.rmsNorm(output)

            return tf.add(inputs, output)
        })
    }

    computeOutputShape(inputShape) {
        const outputShape = this.returnSequences
            ? [inputShape[0], inputShape[1], this.units]
            : [inputShape[0], this.units]
        return outputShape
    }

    getConfig() {
        return {
            ...super.getConfig(),
            units: this.units,
            innerDim: this.innerDim,
            returnSequences: this.returnSequences,
            decayFactor: this.decayFactor,
            activation: this.activation,
            beta: this.beta
        }
    }
}

tf.serialization.registerClass(StateSpace)
