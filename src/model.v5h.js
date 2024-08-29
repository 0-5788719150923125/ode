import ODE from './model.v2.js'

/**
 * A Soft Merging of Experts with Adaptive Routing.
 * @extends ODE
 */
export default class OmnipotentDeterministicEnsemble extends ODE {
    constructor(config) {
        const defaults = {
            layers: 5,
            units: 128,
            embeddings: 512,
            rank: 96,
            numExperts: 3,
            routerDim: 512,
            numHeads: 8,
            headDim: 256,
            headFeatures: 64,
            mlpDim: 512,
            learningRate: 1e-4,
            minLearningRate: 1e-6,
            weightDecay: 1e-5,
            cosineSteps: 2048,
            ALiBiLength: 1024
        }
        super({ ...defaults, ...config })
    }

    defineTokenizer() {
        return this.ode.tokenizers.TokenMonster({
            model: 'englishcode-8000-consistent-v1'
        })
    }

    defineBuild() {
        const inputs = this.ode.layers.input({
            shape: [null]
        })

        const embeddings = this.ode.layers.SharedEmbedding({
            inputDim: this.tokenizer.getLength(),
            outputDim: this.config.embeddings,
            embeddingsInitializer: 'glorotUniform'
        })

        let outputs = embeddings.apply(inputs)

        outputs = this.ode.layers
            .LowRankFactorization({
                outputDim: this.config.units,
                rank: this.config.rank
            })
            .apply(outputs)

        for (let i = 0; i < this.config.layers; i++) {
            outputs = this.ode.layers
                .ProjectedFeatureAttention({
                    numHeads: this.config.numHeads,
                    headDim: this.config.headDim,
                    headFeatures: this.config.headFeatures,
                    ALiBiLength: this.config.ALiBiLength
                })
                .apply(outputs)

            outputs = this.ode.layers
                .SoftMergingOfExpertsMLP({
                    activation: 'mish',
                    routerActivation: 'swish',
                    routerDim: this.config.routerDim,
                    numExperts: this.config.numExperts,
                    expertDim: this.config.mlpDim
                })
                .apply(outputs)
        }

        outputs = this.ode.layers
            .dense({
                units: this.config.embeddings
            })
            .apply(outputs)

        outputs = embeddings.apply(outputs)

        return this.tf.model({ inputs, outputs })
    }

    defineOptimizers() {
        return [
            this.ode.optimizers.Lion({
                learningRate: this.config.learningRate,
                weightDecay: this.config.weightDecay
            })
        ]
    }
}
