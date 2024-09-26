import ODE from './model.v2.js'

/**
 * A baseline, highly-performant small model.
 * @extends ODE
 */
export default class OmniscientDeterministicEngine extends ODE {
    constructor(config) {
        super({
            layers: 6,
            units: 180,
            embeddings: 540,
            numHeads: 4,
            queriesPerHead: 2,
            headDim: 45,
            mlpDim: 1080,
            useBias: true,
            ALiBiLength: 1024,
            learningRate: 1e-3,
            minLearningRate: 1e-5,
            weightDecay: 1e-2,
            cosineSteps: 4096,
            warmupSteps: 128,
            ...config
        })
    }

    defineTokenizer() {
        return this.ode.tokenizers.TokenMonster({
            model: 'englishcode-4096-consistent-v1'
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
                embeddingsInitializer: this.ode.initializers.glorotUniform()
            })
            .apply(inputs)

        outputs = this.defineReductionLayer().apply(outputs)

        for (let i = 0; i < this.config.layers; i++) {
            let normalized = this.ode.layers
                .RMSNorm({ elementwiseAffine: true, useBias: false })
                .apply(outputs)
            const attnOutputs = this.defineAttentionLayer().apply(normalized)
            outputs = this.ode.layers
                .ResidualConnection()
                .apply([attnOutputs, outputs])
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

        return this.tf.model({ inputs, outputs })
    }

    defineReductionLayer() {
        return this.ode.layers.LowRankFactorization({
            units: this.config.units,
            rank: this.config.headDim
        })
    }

    defineAttentionLayer() {
        return this.ode.layers.MultiHeadAttention({
            numHeads: this.config.numHeads,
            headDim: this.config.headDim,
            queriesPerHead: this.config.queriesPerHead,
            ALiBiLength: this.config.ALiBiLength,
            useBias: this.config.useBias
        })
    }

    defineFeedforwardLayer() {
        return this.ode.layers.GatedLinearMLP({
            activation: 'mish',
            gateActivation: 'swish',
            hiddenDim: this.config.mlpDim,
            useBias: this.config.useBias
        })
    }

    defineSchedulers() {
        return [
            this.ode.schedulers.CosineWithRestartsScheduler({
                min: this.config.minLearningRate,
                max: this.config.learningRate,
                totalSteps: this.config.cosineSteps,
                warmupSteps: this.config.warmupSteps
            })
        ]
    }

    defineOptimizers() {
        return [
            this.ode.optimizers.AdamW({
                learningRate: this.config.learningRate,
                weightDecay: this.config.weightDecay
            })
        ]
    }
}
