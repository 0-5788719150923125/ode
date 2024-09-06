import * as tf from '@tensorflow/tfjs'
import LayerBase from './_base.js'

export default class CapsNet extends LayerBase {
    constructor(config) {
        super(config)
        this.units = config?.units || 256
        this.innerDim = config?.innerDim || 1024
        this.dropout = config?.dropout || 0
        this.numCapsules = config?.numCapsules || 8
        this.capsuleDim = config?.capsuleDim || 16
        this.routingIterations = config?.routingIterations || 3
        this.activation = config?.activation || 'relu'
        this.supportsMasking = true
    }

    build(inputShape) {
        // Initialize dense layers for projection
        this.inProjKernel = this.addWeight(
            'inProjKernel',
            [this.units, this.innerDim],
            'float32',
            this.initializers.glorotNormal()
        )
        this.inProjBias = this.addWeight(
            'inProjBias',
            [this.innerDim],
            'float32',
            this.initializers.zeros()
        )

        this.outProjKernel = this.addWeight(
            'outProjKernel',
            [this.numCapsules * this.capsuleDim, this.units],
            'float32',
            this.initializers.glorotNormal()
        )
        this.outProjBias = this.addWeight(
            'outProjBias',
            [this.units],
            'float32',
            this.initializers.zeros()
        )

        // Initialize weights for primary capsules
        this.primaryCapsKernel = this.addWeight(
            'primaryCapsKernel',
            [this.innerDim, this.numCapsules * this.capsuleDim],
            'float32',
            this.initializers.glorotNormal()
        )
        this.primaryCapsBias = this.addWeight(
            'primaryCapsBias',
            [this.numCapsules * this.capsuleDim],
            'float32',
            this.initializers.zeros()
        )

        this.digitCaps = new DigitCaps({
            numCapsules: this.numCapsules,
            capsuleDim: this.capsuleDim,
            routingIterations: this.routingIterations
        })
    }

    call(inputs, kwargs, training = false) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs
            const inputShape = inputs.shape
            const batchSize = inputShape[0]
            const sequenceLength = inputShape[1]

            // Expand and contract projection via feedforward layers
            let outputs = this.ops.applyDense(
                inputs,
                this.inProjKernel.read(),
                this.inProjBias.read()
            )
            // Activate inputs
            outputs = tf.layers
                .activation({ activation: this.activation })
                .apply(outputs)

            // Apply primary capsules
            outputs = this.ops.applyDense(
                outputs,
                this.primaryCapsKernel.read(),
                this.primaryCapsBias.read()
            )

            // Reshape for primary capsules
            outputs = tf.reshape(outputs, [
                batchSize * sequenceLength,
                this.numCapsules,
                this.capsuleDim
            ])

            // Apply digit capsules with dynamic routing
            outputs = this.digitCaps.apply(outputs)

            // Reshape back to original sequence shape
            outputs = tf.reshape(outputs, [
                batchSize,
                sequenceLength,
                this.numCapsules * this.capsuleDim
            ])

            outputs = this.ops.applyDense(
                outputs,
                this.outProjKernel.read(),
                this.outProjBias.read()
            )

            // If training, apply residual dropout
            outputs = kwargs['training']
                ? tf.dropout(outputs, this.dropout)
                : outputs
            // Apply skip connection
            return tf.add(inputs, outputs)
        })
    }

    computeOutputShape(inputShape) {
        return inputShape
    }

    getConfig() {
        return {
            units: this.units,
            innerDim: this.innerDim,
            dropout: this.dropout,
            numCapsules: this.numCapsules,
            capsuleDim: this.capsuleDim,
            activation: this.activation,
            routingIterations: this.routingIterations
        }
    }
}

class DigitCaps extends LayerBase {
    constructor(config) {
        super(config)
        this.numCapsules = config.numCapsules
        this.capsuleDim = config.capsuleDim
        this.routingIterations = config.routingIterations || 3
    }

    build(inputShape) {
        this.W = this.addWeight(
            'capsules',
            [
                1,
                inputShape[1],
                this.numCapsules,
                this.capsuleDim,
                inputShape[2]
            ],
            'float32',
            this.initializers.glorotNormal()
        )
        this.built = true
    }

    call(inputs) {
        return tf.tidy(() => {
            const [batchSize, numPrimaryCaps, primaryCapsDim] = inputs.shape
            const uji = tf.tile(tf.expandDims(inputs, 2), [
                1,
                1,
                this.numCapsules,
                1
            ])
            const ujiHat = tf.sum(
                tf.mul(this.W.read(), tf.expandDims(uji, 3)),
                4
            )

            let b = tf.zeros([batchSize, numPrimaryCaps, this.numCapsules])
            let v = null
            for (let i = 0; i < this.routingIterations; i++) {
                const c = tf.softmax(b, 2)
                const s = tf.sum(tf.mul(tf.expandDims(c, 3), ujiHat), 1)
                v = this.squash(s)
                const agreement = tf.sum(tf.mul(ujiHat, tf.expandDims(v, 1)), 3)
                b = tf.add(b, agreement)
            }

            return v
        })
    }

    squash(s) {
        const squaredNorm = tf.sum(tf.square(s), -1, true)
        const squashFactor = tf.div(squaredNorm, tf.add(1, squaredNorm))
        return tf.mul(squashFactor, tf.div(s, tf.sqrt(squaredNorm)))
    }

    computeOutputShape(inputShape) {
        return [inputShape[0], this.numCapsules, this.capsuleDim]
    }

    getConfig() {
        const config = {
            numCapsules: this.numCapsules,
            capsuleDim: this.capsuleDim,
            routingIterations: this.routingIterations
        }
        return config
    }
}

tf.serialization.registerClass(CapsNet)
tf.serialization.registerClass(DigitCaps)
