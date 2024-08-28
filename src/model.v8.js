import ODE from './model.v8.js'

/**
 * In development.
 * @extends ODE
 */
export default class OptimalDecisionEngine extends ODE {
    defineLossFunction() {
        return {
            name: 'MiLeCrossEntropy',
            smoothing: 0.0001
        }
    }
}
