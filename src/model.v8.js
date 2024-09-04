import ODE from './model.v2.js'

/**
 * A model used for active research and development.
 * @extends ODE
 */
export default class OmnilateralDynamicEvaluator extends ODE {
    constructor(config) {
        const defaults = {
            layers: 6,
            units: 180,
            embeddings: 540,
            numHeads: 4,
            queriesPerHead: 2,
            headDim: 45,
            mlpDim: 1080,
            useBias: true,
            ALiBiLength: 1024,
            learningRate: 1e-4,
            minLearningRate: 1e-6,
            weightDecay: 1e-5,
            cosineSteps: 4096,
            warmupSteps: 128
        }
        super({ ...defaults, ...config })
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
            .ParabolicCompression({
                units: this.config.units,
                numSteps: 4
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

            outputs = this.ode.layers
                .PrimerAttention({
                    numHeads: this.config.numHeads,
                    headDim: this.config.headDim,
                    queriesPerHead: this.config.queriesPerHead,
                    ALiBiLength: this.config.ALiBiLength,
                    useBias: this.config.useBias
                })
                .apply(outputs)

            outputs = this.ode.layers
                .GatedLinearMLP({
                    activation: 'mish',
                    gateActivation: 'swish',
                    hiddenDim: this.config.mlpDim,
                    useBias: this.config.useBias
                })
                .apply(outputs)
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

    defineSchedulers() {
        return [
            this.ode.schedulers.cosineWithRestartsScheduler(
                this.config.minLearningRate,
                this.config.learningRate,
                this.config.cosineSteps,
                this.config.warmupSteps
            )
        ]
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
