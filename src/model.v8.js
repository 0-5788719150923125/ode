import ODE from './model.v7.js'

/**
 * A baseline, highly-performant small model.
 * @extends ODE
 */
export default class OmniscientDeterministicEngine extends ODE {
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

        outputs = this.ode.layers
            .ParabolicCompression({
                units: this.config.units,
                numSteps: 4
            })
            .apply(outputs)

        const exportedStates = []

        for (let i = 0; i < this.config.layers; i++) {
            outputs = this.ode.layers
                .PrimerAttention({
                    numHeads: this.config.numHeads,
                    headDim: this.config.headDim,
                    queriesPerHead: this.config.queriesPerHead,
                    ALiBiLength: this.config.ALiBiLength,
                    useBias: this.config.useBias
                })
                .apply(outputs)

            exportedStates.push(outputs)

            outputs = this.ode.layers
                .GatedLinearMLP({
                    activation: 'mish',
                    gateActivation: 'swish',
                    hiddenDim: this.config.mlpDim,
                    useBias: this.config.useBias
                })
                .apply(outputs)

            exportedStates.push(outputs)
        }

        outputs = this.ode.layers
            .dense({
                units: this.tokenizer.getLength(),
                kernelInitializer: this.ode.initializers.glorotUniform()
            })
            .apply(outputs)

        return this.tf.model({ inputs, outputs: [outputs, ...exportedStates] })
    }
}
