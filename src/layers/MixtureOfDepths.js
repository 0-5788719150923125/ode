import * as tf from '@tensorflow/tfjs'
import LayerBase from './base.js'

// https://arxiv.org/abs/2404.02258
export default class MixtureOfDepths extends LayerBase {
    constructor(config) {
        super(config)
        this.experts = config.experts || []
        this.numExperts = config.numExperts || this.experts.length
        this.capacity = config.capacity || 0.125
        this.temperature = config.temperature || 0.1
        this.auxLossWeight = config.auxLossWeight || 0.01
        this.emaDecay = config.emaDecay || 0.99
        this.expertUsageEMA = null
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
                mode: 'fanAvg'
            })
        )
        this.routerBias = this.addWeight(
            'routerBias',
            [1],
            'float32',
            tf.initializers.zeros()
        )
        this.expertUsageEMA = tf.variable(tf.zeros([this.numExperts]), false)
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

            return tf.customGrad((x, save) => {
                // Forward pass: Top-k selection
                const k = Math.floor(this.capacity * timeSteps)
                const { values: topKValues, indices: topKIndices } = tf.topk(
                    routerLogits,
                    k
                )

                const topkMask = tf
                    .oneHot(topKIndices, timeSteps)
                    .sum(1)
                    .expandDims(-1)

                // Apply top-k mask to inputs
                const selectedTokens = x.mul(topkMask)
                const residualTokens = x.mul(
                    tf.onesLike(topkMask).sub(topkMask)
                )

                // Apply layer to routed tokens
                let selectedOutputs = selectedTokens
                for (const expert of this.experts) {
                    selectedOutputs = expert.apply(selectedOutputs)
                }

                // Combine processed tokens with residual tokens
                const output = selectedOutputs.add(residualTokens)

                const savedTensors = [routerLogits, topkMask, x]

                // Compute auxiliary loss
                if (kwargs.training) {
                    savedTensors.push(
                        this.computeAuxLoss(topKValues, topKIndices)
                    )
                }

                save(savedTensors)

                // Define gradient function
                const gradFunc = (dy, saved) => {
                    const [routerLogits, topkMask, originalInputs] =
                        saved.slice(0, 2)

                    let auxLoss
                    if (kwargs.training) {
                        auxLoss = saved[3]
                    }

                    // Backward pass: Gumbel-Softmax
                    const gumbelMask = this.ode.ops
                        .gumbelSoftmax(routerLogits, this.temperature)
                        .expandDims(-1)

                    // Compute gradients for the selected tokens
                    let selectedGrads = dy.mul(gumbelMask)
                    for (const expert of this.experts) {
                        selectedGrads = expert.apply(selectedGrads)
                    }

                    // Compute gradients for the residual tokens
                    const residualGrads = dy.mul(
                        tf.onesLike(gumbelMask).sub(gumbelMask)
                    )

                    // Combine the selected and residual gradients
                    let inputGrads = selectedGrads.add(residualGrads)

                    // Add auxiliary loss gradient
                    if (kwargs.training) {
                        inputGrads = inputGrads.add(
                            auxLoss.mul(this.auxLossWeight)
                        )
                    }

                    return inputGrads
                }

                return { value: output, gradFunc }
            })(inputs)
        })
    }

    computeAuxLoss(topKIndices) {
        return tf.tidy(() => {
            const [batchSize, k] = topKIndices.shape
            const numExperts = this.numExperts

            // Compute current expert usage
            const currentUsage = tf
                .oneHot(topKIndices.cast('int32'), numExperts)
                .sum([0, 1])
                .div(tf.scalar(batchSize * k))

            // Update EMA
            const newEMA = this.expertUsageEMA
                .mul(this.emaDecay)
                .add(currentUsage.mul(1 - this.emaDecay))
            this.expertUsageEMA.assign(newEMA)

            // Compute load balancing loss
            const idealUsage = tf.ones([numExperts]).div(numExperts)
            const loadBalancingLoss = tf
                .squaredDifference(this.expertUsageEMA, idealUsage)
                .mean()

            // Compute expert utilization loss
            const utilizationLoss = tf
                .log(this.expertUsageEMA.add(1e-5))
                .neg()
                .mean()

            // Combine losses
            const loss = loadBalancingLoss.add(utilizationLoss)
            // console.log(loss.dataSync()[0])
            return loss
        })
    }

    getConfig() {
        return {
            ...super.getConfig(),
            numExperts: this.numExperts,
            capacity: this.capacity,
            temperature: this.temperature,
            auxLossWeight: this.auxLossWeight,
            emaDecay: this.emaDecay
        }
    }
}

tf.serialization.registerClass(MixtureOfDepths)
