import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

export default class Autoencoder extends LayerBase {
    constructor(config) {
        super(config)
        this.innerDim = config?.innerDim || 1024
        this.bottleneck = config?.bottleneck || 128
        this.outputDim = config?.outputDim || 256
        this.encoderActivation = config?.encoderActivation || 'relu'
        this.decoderActivation = config?.decoderActivation || 'relu'
        this.variational = config?.variational || false
        this.beta = config?.beta || 1.0
    }

    build(inputShape) {
        const inputDim = inputShape[inputShape.length - 1]
        const multiplier = this.variational ? 2 : 1

        // Initialize dense layers for encoder
        this.encoderKernel1 = this.addWeight(
            'encoderKernel1',
            [inputDim, this.innerDim],
            'float32',
            tf.initializers.glorotNormal(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.encoderBias1 = this.addWeight(
            'encoderBias1',
            [this.innerDim],
            'float32',
            tf.initializers.zeros(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.encoderKernel2 = this.addWeight(
            'encoderKernel2',
            [this.innerDim, this.bottleneck * multiplier],
            'float32',
            tf.initializers.glorotNormal(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.encoderBias2 = this.addWeight(
            'encoderBias2',
            [this.bottleneck * multiplier],
            'float32',
            tf.initializers.zeros(),
            tf.regularizers.l2({ l2: 0.01 })
        )

        // Initialize dense layers for decoder
        this.decoderKernel1 = this.addWeight(
            'decoderKernel1',
            [this.bottleneck, this.innerDim],
            'float32',
            tf.initializers.glorotNormal(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.decoderBias1 = this.addWeight(
            'decoderBias1',
            [this.innerDim],
            'float32',
            tf.initializers.zeros(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.decoderKernel2 = this.addWeight(
            'decoderKernel2',
            [this.innerDim, this.outputDim],
            'float32',
            tf.initializers.glorotNormal(),
            tf.regularizers.l2({ l2: 0.01 })
        )
        this.decoderBias2 = this.addWeight(
            'decoderBias2',
            [this.outputDim],
            'float32',
            tf.initializers.zeros(),
            tf.regularizers.l2({ l2: 0.01 })
        )
    }

    computeVariance(inputs, kwargs) {
        // Compute the mean and log-variance
        const [mean, logVar] = tf.split(inputs, 2, -1)
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
            this.extraLoss = tf.keep(klDivergence)
        }

        // Sample from the latent space using the reparameterization trick
        const epsilon = tf.randomNormal(mean.shape)
        return mean.add(epsilon.mul(expLogVar.sqrt()))
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            // Encode the inputs to the bottleneck representation
            let outputs = this.ops.applyDense(
                inputs,
                this.encoderKernel1.read(),
                this.encoderBias1.read()
            )

            outputs = tf.layers
                .activation({ activation: this.encoderActivation })
                .apply(outputs)

            outputs = this.ops.applyDense(
                outputs,
                this.encoderKernel2.read(),
                this.encoderBias2.read()
            )

            if (this.variational) {
                outputs = this.computeVariance(outputs, kwargs)
            }

            // Decode the bottleneck representation to the output dimensionality
            outputs = this.ops.applyDense(
                outputs,
                this.decoderKernel1.read(),
                this.decoderBias1.read()
            )

            outputs = tf.layers
                .activation({ activation: this.decoderActivation })
                .apply(outputs)

            outputs = this.ops.applyDense(
                outputs,
                this.decoderKernel2.read(),
                this.decoderBias2.read()
            )

            return outputs
        })
    }

    computeOutputShape(inputShape) {
        return [inputShape[0], inputShape[1], this.outputDim]
    }

    getConfig() {
        return {
            ...super.getConfig(),
            innerDim: this.innerDim,
            bottleneck: this.bottleneck,
            outputDim: this.outputDim,
            encoderActivation: this.encoderActivation,
            decoderActivation: this.decoderActivation,
            variational: this.variational,
            beta: this.beta
        }
    }
}

tf.serialization.registerClass(Autoencoder)
