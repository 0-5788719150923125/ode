// https://github.com/lhallee/Multi_Head_Mixture_of_Experts__MH-MOE/blob/main/mhmoe.py
class MultiHeadMoeBlock extends LayerBase {
    constructor(config) {
        super(config)
        this.hiddenDim = config.hiddenDim || 64
        this.numExperts = config.numExperts || 4
        this.numHeads = config.numHeads || 4
        this.topk = config.topk || 2
        this.headDim = this.hiddenDim / this.numHeads
        this.roundedDim =
            Math.floor(this.hiddenDim / this.numHeads) * this.numHeads
    }

    build(inputShape) {
        this.multiHeadLayer = tf.layers.dense({
            units: this.roundedDim,
            useBias: false,
            activation: 'linear',
            kernelInitializer: 'glorotUniform'
        })

        this.router = new MHRouter({
            numExperts: this.numExperts,
            hiddenDim: this.hiddenDim,
            numHeads: this.numHeads
        })

        this.experts = []
        for (let i = 0; i < this.numExperts; i++) {
            const expert = tf.layers.dense({
                units: this.headDim,
                useBias: false,
                activation: 'linear',
                kernelInitializer: 'glorotUniform'
            })
            this.experts.push(expert)
        }

        this.mergeLayer = tf.layers.dense({
            units: this.hiddenDim,
            useBias: false,
            activation: 'linear',
            kernelInitializer: 'glorotUniform'
        })
    }

    call(inputs) {
        return tf.tidy(() => {
            inputs = Array.isArray(inputs) ? inputs[0] : inputs

            const batchSize = inputs.shape[0]
            const seqLen = inputs.shape[1]

            // Project inputs to rounded dimension
            let x = this.multiHeadLayer.apply(inputs)
            x = x.reshape([batchSize * seqLen * this.numHeads, this.headDim])

            // Router
            const routerLogits = this.router.apply(x)
            const routerWeights = routerLogits.softmax()
            const topkOutputs = tf.topk(routerWeights, this.topk)

            // Call experts densely, faster than selective loops
            const expertOutputs = []
            for (const expert of this.experts) {
                expertOutputs.push(expert.apply(x))
            }
            const expertStack = tf.stack(expertOutputs, 1)

            // Select top-k expert outputs
            const batchIndices = tf.range(0, expertStack.shape[0]).expandDims(1)
            const gatherIndices = tf.concat(
                [batchIndices.cast('int32'), topkOutputs.indices.cast('int32')],
                1
            )
            const selectedExpertOutputs = tf.gatherND(
                expertStack.cast('float32'),
                gatherIndices.cast('int32')
            )

            // Multiply selected expert outputs with router weights elementwise
            const weightedExpertOutputs = selectedExpertOutputs.mul(
                topkOutputs.values.expandDims(-1)
            )

            // Combine top-k expert outputs
            x = weightedExpertOutputs.sum(1)

            // Back to original shape
            x = x.reshape([batchSize, seqLen, this.headDim])
            x = this.mergeLayer.apply(x)

            return x
        })
    }

    getConfig() {
        return {
            ...super.getConfig(),
            hiddenDim: this.hiddenDim,
            numExperts: this.numExperts,
            numHeads: this.numHeads,
            topk: this.topk
        }
    }
}

class MHRouter extends LayerBase {
    constructor(config) {
        super({ name: `mh-router-${randomString()}`, ...config })
        this.numExperts = config.numExperts
        this.hiddenDim = config.hiddenDim
        this.numHeads = config.numHeads
    }

    build(inputShape) {
        this.expertEmbedding = this.addWeight(
            'expertEmbedding',
            [this.hiddenDim / this.numHeads, this.numExperts],
            'float32',
            tf.initializers.randomNormal({ mean: 0, stddev: 1 })
        )
    }

    call(inputs) {
        return tf.matMul(inputs, this.expertEmbedding.read())
    }

    getWeights() {
        return [this.expertEmbedding.read()]
    }

    setWeights(weights) {
        this.expertEmbedding.write(weights[0])
    }

    getConfig() {
        return {
            ...super.getConfig(),
            numExperts: this.numExperts,
            hiddenDim: this.hiddenDim,
            numHeads: this.numHeads
        }
    }
}
