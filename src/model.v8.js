import ODE from './model.v6.js'

/**
 * A model used for active research and development.
 * @extends ODE
 */
export default class OpenDerivationExperiment extends ODE {
    constructor(config) {
        super(config)
        this.config.headFeatures = 23
    }

    defineAttentionLayer() {
        return this.ode.layers.ProjectedFeatureAttention({
            numHeads: this.config.numHeads,
            headDim: this.config.headDim,
            headFeatures: this.config.headFeatures,
            queriesPerHead: this.config.queriesPerHead,
            ALiBiLength: this.config.ALiBiLength,
            useBias: this.config.useBias
        })
    }
}
