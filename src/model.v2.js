import ODE from './model.v0.js'

/**
 * A GRU-based RNN that uses a time-distributed, dense output
 * layer. This is quite different from common RNNs, in that it functions more
 * like a sequence-to-sequence model. Rather than training on a single label,
 * this model trains on vectors of them, shifted by one to the right.
 * @extends ODE
 */
export default class OmnipresentDiabolicalErudite extends ODE {
    constructor(config) {
        super(config)
        this.layers = config.layers || 3
        this.units = config.units || 64
        this.epsilon = config.epsilon || 1e-5
    }

    defineBuild() {
        const inputs = this.tf.input({ shape: [null] })
        let outputs = this.tf.layers
            .embedding({
                inputDim: this.tokenizer.getLength(),
                outputDim: this.units,
                embeddingsInitializer: 'glorotUniform'
            })
            .apply(inputs)

        outputs = this.tf.layers
            .layerNormalization({ epsilon: this.epsilon })
            .apply(outputs)

        for (let i = 0; i < this.layers; i++) {
            outputs = this.tf.layers
                .gru({
                    units: this.units,
                    activation: 'softsign',
                    kernelInitializer: 'glorotUniform',
                    recurrentActivation: 'sigmoid',
                    recurrentInitializer: 'orthogonal',
                    returnSequences: true
                })
                .apply(outputs)

            outputs = this.tf.layers
                .layerNormalization({ epsilon: this.epsilon })
                .apply(outputs)
        }

        outputs = this.tf.layers
            .timeDistributed({
                layer: this.tf.layers.dense({
                    units: this.tokenizer.getLength(),
                    activation: 'linear'
                })
            })
            .apply(outputs)

        this.model = this.tf.model({ inputs, outputs })
    }

    defineOptimizers() {
        this.optimizers = [
            this.ode.optimizers.AdamW({
                learningRate: this.config.learningRate || 1e-3,
                beta1: this.config.beta1 || 0.9,
                beta2: this.config.beta2 || 0.999,
                epsilon: this.config.epsilon || 1e-7,
                weightDecay: this.config.weightDecay || 1e-1
            })
        ]
    }

    defineSchedulers() {
        const initialLr = 0.000333
        const peakLr = 0.00333
        const iterations = 333
        const modulation = 0.666
        this.optimizers[0].learningRate = initialLr
        this.schedulers = [
            this.ode.schedulers.cosineScheduler(
                initialLr,
                peakLr,
                iterations,
                modulation
            )
        ]
    }
}
