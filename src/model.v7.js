import ODE from './model.v6.js'

/**
 * A model used for active research and development.
 * @extends ODE
 */
export default class OptionalDecisionEvaluator extends ODE {
    constructor(config) {
        super({
            learningRate: 1e-3,
            weightDecay: 0.01,
            selfModeling: {
                filters: 1080,
                auxLossFunction: 'meanSquaredError',
                auxiliaryWeight: 10.0,
                kernelSize: 3,
                strides: 1,
                activation: 'selu'
            },
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

        const modeler = this.ode.layers.SelfModeling({
            prefix: 'mod',
            filters: this.config.selfModeling.filters,
            kernelSize: this.config.selfModeling.kernelSize,
            strides: this.config.selfModeling.strides,
            activation: this.config.selfModeling.activation,
            useBias: true
        })

        const hiddenStates = []

        for (let i = 0; i < this.config.layers; i++) {
            let normalized = this.ode.layers
                .RMSNorm({ elementwiseAffine: true, useBias: false })
                .apply(outputs)
            const attnOutput = this.defineAttentionLayer().apply(normalized)
            const predictedOutput = modeler.apply(normalized)
            hiddenStates.push(...[attnOutput, predictedOutput])
            outputs = this.ode.layers
                .ResidualConnection()
                .apply([attnOutput, outputs])
            normalized = this.ode.layers
                .RMSNorm({ elementwiseAffine: true, useBias: false })
                .apply(outputs)
            const ffdOutput = this.defineFeedforwardLayer().apply(normalized)
            outputs = this.ode.layers
                .ResidualConnection()
                .apply([ffdOutput, outputs])
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
            outputs: [outputs, ...hiddenStates]
        })
    }

    postProcessing(outputs) {
        const selfModelingLoss = this.modelSelf(
            outputs.slice(1),
            this.config.selfModeling.auxLossFunction,
            this.config.selfModeling.auxiliaryWeight
        )
        return selfModelingLoss
    }

    // https://arxiv.org/abs/2407.10188
    modelSelf(
        hiddenStates,
        auxLossFunction = 'meanSquaredError',
        auxiliaryWeight = 1.0
    ) {
        let loss = this.tf.scalar(0)

        hiddenStates.map((hiddenState, i) => {
            if (i % 2 === 0) return
            const actual = hiddenStates[i - 1]
            const prediction = hiddenState
            const ls = this.ode.losses[auxLossFunction](
                actual,
                prediction
                // -1,
                // null,
                // this.tf.Reduction.MEAN
            )
            loss = loss.add(ls)
        })
        const scaledLoss = this.tf.mul(loss, this.tf.scalar(auxiliaryWeight))
        // console.log(scaledLoss.dataSync())
        return scaledLoss
    }
}
