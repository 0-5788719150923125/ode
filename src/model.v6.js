import ODE from './model.v2.js'

/**
 * In development.
 * @extends ODE
 */
export default class OpenDoorExperiment extends ODE {
    constructor(config) {
        const defaults = {
            layers: 4,
            units: 256,
            numHeads: 8,
            queriesPerHead: 1,
            headDim: 128,
            mlpDim: 1024,
            useBias: true,
            ALiBiLength: 1024,
            learningRate: 1e-4,
            weightDecay: 1e-5
        }
        super({ ...defaults, ...config })
    }

    defineTokenizer() {
        return this.ode.tokenizers.TokenMonster({
            model: 'englishcode-8000-clean-v1'
        })
    }

    defineBuild() {
        const inputs = this.ode.layers.input({
            shape: [null]
        })

        const embeddings = this.ode.layers.SharedEmbedding({
            inputDim: this.tokenizer.getLength(),
            outputDim: this.config.units,
            embeddingsInitializer: 'glorotUniform'
        })

        let outputs = embeddings.apply(inputs)

        for (let i = 0; i < this.config.layers; i++) {
            outputs = this.ode.layers
                .MultiHeadAttention({
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

        outputs = embeddings.apply(outputs)

        return this.tf.model({ inputs, outputs })
    }

    defineSchedulers() {
        return [this.ode.schedulers.constantScheduler(this.config.learningRate)]
    }

    defineOptimizers() {
        return [
            this.ode.optimizers.Lion({
                learningRate: this.config.learningRate,
                weightDecay: this.config.weightDecay,
                useGc: true
            })
        ]
    }
}
