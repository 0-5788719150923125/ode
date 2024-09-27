import ODE from './model.v6.js'

/**
 * A model used for active research and development.
 * @extends ODE
 */
export default class OptionalDecisionExecution extends ODE {
    constructor(config) {
        super({
            learningRate: 1e-3,
            weightDecay: 0.01,
            selfModel: true,
            auxLossFunction: 'hingeLoss',
            auxiliaryWeight: 1.0,
            ...config
        })
    }

    defineSchedulers() {
        return [
            this.ode.schedulers.ConstantScheduler({
                max: this.config.learningRate,
                warmupSteps: this.config.warmupSteps
            })
        ]
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

        // Self-modeling layer
        const modeler = this.ode.layers.dense({
            prefix: 'mod',
            units: this.config.units,
            kernelInitializer: this.ode.initializers.glorotUniform()
        })

        this.hiddenStates = []

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
            const actualOutput = this.defineFeedforwardLayer().apply(normalized)
            const predictedOutput = modeler.apply(normalized)
            this.hiddenStates.push(actualOutput)
            this.hiddenStates.push(predictedOutput)
            outputs = this.ode.layers
                .ResidualConnection()
                .apply([actualOutput, outputs])
        }

        outputs = this.ode.layers
            .dense({
                prefix: 'head',
                units: this.tokenizer.getLength(),
                kernelInitializer: this.ode.initializers.glorotUniform()
            })
            .apply(outputs)

        return this.tf.model({
            inputs,
            outputs: [outputs, ...this.hiddenStates]
        })
    }

    postProcessing(logits) {
        const hiddenStates = logits.slice(1)
        const selfModelingLoss = this.modelSelf(
            hiddenStates,
            this.config.auxLossFunction,
            this.config.auxiliaryWeight
        )
        return selfModelingLoss
    }

    // https://arxiv.org/abs/2407.10188
    modelSelf(
        hiddenStates,
        auxLossFunction = 'hingeLoss',
        auxiliaryWeight = 0.1
    ) {
        return this.tf.tidy(() => {
            let loss = this.tf.scalar(0)

            hiddenStates.map((hiddenState, i) => {
                if (i % 2 === 0) return
                const actual = hiddenStates[i - 1]
                const prediction = hiddenState
                // const ls = this.tf.losses.cosineDistance(actual, prediction, 0)
                // loss = loss.add(this.tf.clipByValue(ls, 0, 2))
                const ls = this.tf.losses[auxLossFunction](actual, prediction)
                // console.log(ls.dataSync())
                loss = loss.add(this.tf.clipByValue(ls, 0, 2))
            })
            return this.tf.mul(loss, this.tf.scalar(auxiliaryWeight))
        })
    }
}
