import ODE from './model.v0.js'

/**
 * A baseline, highly-performant small model.
 * @extends ODE
 */
export default class OmniscientDeterministicEngine extends ODE {
    constructor(config) {
        const defaults = {
            layers: 6,
            units: 180,
            embeddings: 540,
            numHeads: 4,
            queriesPerHead: 2,
            headDim: 45,
            mlpDim: 1080,
            useBias: true,
            ALiBiLength: 1024,
            learningRate: 1e-4,
            minLearningRate: 1e-6,
            weightDecay: 1e-5,
            cosineSteps: 4096,
            warmupSteps: 128
        }
        super({ ...defaults, ...config })
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

        outputs = this.ode.layers
            .ParabolicCompression({
                units: this.config.units,
                numSteps: 4
            })
            .apply(outputs)

        const exportedStates = []

        for (let i = 0; i < this.config.layers; i++) {
            outputs = this.ode.layers
                .PrimerAttention({
                    numHeads: this.config.numHeads,
                    headDim: this.config.headDim,
                    queriesPerHead: this.config.queriesPerHead,
                    ALiBiLength: this.config.ALiBiLength,
                    useBias: this.config.useBias
                })
                .apply(outputs)

            exportedStates.push(outputs)

            outputs = this.ode.layers
                .GatedLinearMLP({
                    activation: 'mish',
                    gateActivation: 'swish',
                    hiddenDim: this.config.mlpDim,
                    useBias: this.config.useBias
                })
                .apply(outputs)

            exportedStates.push(outputs)
        }

        outputs = this.ode.layers
            .dense({
                units: this.tokenizer.getLength(),
                kernelInitializer: this.ode.initializers.glorotUniform()
            })
            .apply(outputs)

        return this.tf.model({ inputs, outputs: [outputs, ...exportedStates] })
    }

    defineLossFunction() {
        return {
            name: 'softmaxCrossEntropy',
            smoothing: 0.0001,
            reduction: this.tf.Reduction.MEAN
        }
    }

    defineSchedulers() {
        return [
            this.ode.schedulers.cosineWithRestartsScheduler(
                this.config.minLearningRate,
                this.config.learningRate,
                this.config.cosineSteps,
                this.config.warmupSteps
            )
        ]
    }

    defineOptimizers() {
        return [
            this.ode.optimizers.Lion({
                learningRate: this.config.learningRate,
                weightDecay: this.config.weightDecay,
                useGc: true,
                adaNorm: true
            })
        ]
    }
}
