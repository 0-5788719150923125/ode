import ODE from './model.v6.js'

/**
 * A model used for active research and development.
 * @extends ODE
 */
export default class OptionalDecisionExecution extends ODE {
    constructor(config) {
        super({ selfModel: true, auxiliaryWeight: 1.0, ...config })
    }

    defineSchedulers() {
        return [
            this.ode.schedulers.ConstantScheduler({
                max: this.config.learningRate,
                warmupSteps: this.config.warmupSteps
            })
        ]
    }

    defineBuild() {
        const inputs = this.ode.layers.input({
            shape: [null]
        })

        let outputs = this.ode.layers
            .embedding({
                inputDim: this.tokenizer.getLength(),
                outputDim: this.config.embeddings,
                embeddingsInitializer: this.ode.initializers.glorotUniform()
            })
            .apply(inputs)

        outputs = this.defineReductionLayer().apply(outputs)

        // Self-modeling layer
        const modeler = this.ode.layers.dense({
            prefix: 'mod',
            units: this.config.units,
            kernelInitializer: this.ode.initializers.glorotUniform()
        })

        const hiddenStates = []

        for (let i = 0; i < this.config.layers; i++) {
            let normalized = this.ode.layers
                .RMSNorm({ elementwiseAffine: true, useBias: false })
                .apply(outputs)
            const actualState = this.defineAttentionLayer().apply(normalized)
            const predictedState = modeler.apply(normalized)
            hiddenStates.push(actualState)
            hiddenStates.push(predictedState)
            outputs = this.ode.layers
                .ResidualConnection()
                .apply([actualState, outputs])
            normalized = this.ode.layers
                .RMSNorm({ elementwiseAffine: true, useBias: false })
                .apply(outputs)
            const ffdOutputs = this.defineFeedforwardLayer().apply(normalized)
            outputs = this.ode.layers
                .ResidualConnection()
                .apply([ffdOutputs, outputs])
        }

        outputs = this.ode.layers
            .dense({
                prefix: 'head',
                units: this.tokenizer.getLength(),
                kernelInitializer: this.ode.initializers.glorotUniform()
            })
            .apply(outputs)

        return this.tf.model({ inputs, outputs: [outputs, ...hiddenStates] })
    }
}
