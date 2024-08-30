import ODE from './model.v6.js'

/**
 * A model used for active development and testing.
 * @extends ODE
 */
export default class OptionalDecisionExecution extends ODE {
    defineAttentionLayer() {
        return this.ode.layers.PrimerAttention({
            numHeads: this.config.numHeads,
            headDim: this.config.headDim,
            queriesPerHead: this.config.queriesPerHead,
            ALiBiLength: this.config.ALiBiLength,
            useBias: this.config.useBias
        })
    }
}
