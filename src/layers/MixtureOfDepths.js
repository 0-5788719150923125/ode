import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

export default class MixtureOfDepths extends LayerBase {
    constructor(config) {
        super(config)
        this.experts = config.experts || []
        this.numExperts = config.numExperts || this.experts.length
        this.capacity = config.capacity || 0.125
    }

    build(inputShape) {
        const inputDim = inputShape[inputShape.length - 1]

        this.routerKernel = this.addWeight(
            'routerKernel',
            [inputDim, 1],
            'float32',
            tf.initializers.varianceScaling({
                scale: 0.01,
                distribution: 'normal',
                mode: 'fanAvg',
                seed: this.ops.getSeed()
            })
        )
        this.routerBias = this.addWeight(
            'routerBias',
            [1],
            'float32',
            tf.initializers.zeros()
        )
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const [batchSize, timeSteps, inputDim] = inputs.shape

            // Router network
            const routerLogits = this.ops
                .applyDense(
                    inputs,
                    this.routerKernel.read(),
                    this.routerBias.read()
                )
                .reshape([batchSize, timeSteps])

            // Straight-Through Estimator (STE) for top-k selection
            const output = tf.customGrad((inputs, save) => {
                const k = Math.floor(this.capacity * timeSteps)
                const { indices: topKIndices } = tf.topk(routerLogits, k)
                const topkMask = tf
                    .oneHot(topKIndices, timeSteps)
                    .sum(1)
                    .expandDims(-1)

                // Apply top-k mask to inputs
                const selectedTokens = inputs.mul(topkMask)
                const residualTokens = inputs.mul(
                    tf.onesLike(topkMask).sub(topkMask)
                )

                // Apply layer to routed tokens
                let selectedOutputs = selectedTokens
                for (const expert of this.experts) {
                    selectedOutputs = expert.apply(selectedOutputs)
                }

                // Combine processed tokens with residual tokens
                const sparseOutputs = selectedOutputs.add(residualTokens)

                save([inputs, routerLogits])

                // return {
                //     value: sparseOutputs,
                //     gradFunc: (dy, saved) => [dy]
                // }

                return {
                    value: sparseOutputs,
                    gradFunc: (dy, saved) => {
                        const [inputs, routerLogits] = saved
                        const softMask = tf.softmax(routerLogits).expandDims(-1)
                        const gradInputs = dy.mul(softMask)
                        return [gradInputs]
                    }
                }
            })(inputs)

            return output
        })
    }

    getConfig() {
        return {
            ...super.getConfig(),
            numExperts: this.numExperts,
            capacity: this.capacity
        }
    }
}

tf.serialization.registerClass(MixtureOfDepths)
