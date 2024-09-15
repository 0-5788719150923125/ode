import ODE from './model.v4.js'

/**
 * A model used for active research and development.
 * @extends ODE
 */
export default class OmnilateralDynamicEvaluator extends ODE {
    constructor(config) {
        super(config)
    }

    defineTokenizer() {
        return this.ode.tokenizers.TokenMonster({
            model: 'englishcode-4096-clean-v1'
        })
    }

    defineBuild() {
        const inputs = this.ode.layers.input({
            shape: [null]
        })

        let outputs = this.ode.layers
            .embedding({
                inputDim: this.tokenizer.getLength(),
                outputDim: this.config.embeddings,
                kernelInitializer: this.ode.initializers.glorotUniform()
            })
            .apply(inputs)

        outputs = this.ode.layers
            .LowRankFactorization({
                units: this.config.units,
                rank: this.config.headDim
            })
            .apply(outputs)

        for (let i = 0; i < this.config.layers; i++) {
            if (i % 2 !== 0) {
                const quarter = this.config.units / 4
                let [updated, retained] = this.ode.layers
                    .Split({
                        axis: -1,
                        numOrSizeSplits: [quarter, quarter * 3]
                    })
                    .apply(outputs)

                updated = this.ode.layers
                    .FastAssociativeMemory({
                        activation: 'gelu',
                        numSteps: 3,
                        learningRate: 0.1,
                        decayRate: 0.9
                    })
                    .apply(updated)

                outputs = this.ode.layers
                    .concatenate({ axis: -1 })
                    .apply([retained, updated])
            }

            let normalized = this.ode.layers
                .RMSNorm({ elementwiseAffine: true, useBias: true })
                .apply(outputs)

            const attnOutputs = this.ode.layers
                .PrimerAttention({
                    numHeads: this.config.numHeads,
                    headDim: this.config.headDim,
                    queriesPerHead: this.config.queriesPerHead,
                    ALiBiLength: this.config.ALiBiLength,
                    useBias: this.config.useBias
                })
                .apply(normalized)

            outputs = this.ode.layers
                .ResidualConnection()
                .apply([attnOutputs, outputs])

            normalized = this.ode.layers
                .RMSNorm({ elementwiseAffine: true, useBias: true })
                .apply(outputs)

            const ffdOutputs = this.ode.layers
                .GatedLinearMLP({
                    activation: 'mish',
                    gateActivation: 'swish',
                    hiddenDim: this.config.mlpDim,
                    useBias: this.config.useBias
                })
                .apply(normalized)

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

        return this.tf.model({ inputs, outputs })
    }

    defineOptimizers() {
        return [
            this.ode.optimizers.Lion({
                learningRate: this.config.learningRate,
                weightDecay: this.config.weightDecay,
                useGc: true,
                adaNorm: true
            })
        ]
    }
}
