import ODE from './model.v6.js'

/**
 * A model used for active research and development.
 * @extends ODE
 */
export default class OptionalDecisionExecution extends ODE {
    constructor(config) {
        const defaults = {
            selfModel: true,
            auxiliaryWeight: 10.0
        }
        super({ ...defaults, ...config })
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
            const actualState = this.defineAttentionLayer().apply(outputs)
            const predictedState = modeler.apply(outputs)
            hiddenStates.push(actualState)
            hiddenStates.push(predictedState)
            outputs = this.defineFeedforwardLayer().apply(actualState)
        }

        outputs = this.ode.layers
            .dense({
                units: this.tokenizer.getLength(),
                kernelInitializer: this.ode.initializers.glorotUniform()
            })
            .apply(outputs)

        return this.tf.model({ inputs, outputs: [outputs, ...hiddenStates] })
    }
}
