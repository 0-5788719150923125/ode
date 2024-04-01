import * as tf from '@tensorflow/tfjs'

class AdamW extends tf.AdamOptimizer {
    constructor(
        learningRate = 0.1,
        beta1 = 0.9,
        beta2 = 0.999,
        epsilon = 1e-7,
        decayRate = 1e-4,
        includeInWeightDecay,
        excludeFromWeightDecay
    ) {
        super(learningRate, beta1, beta2, epsilon)
        this.ENGINE = tf.engine()
        this.decayRate = decayRate
        this.includeInWeightDecay = includeInWeightDecay
        this.excludeFromWeightDecay = excludeFromWeightDecay
    }
    applyGradients(variableGradients) {
        tf.tidy(() => {
            const varNames = Array.isArray(variableGradients)
                ? variableGradients.map((v) => v.name)
                : Object.keys(variableGradients)

            // Apply weight decay
            varNames.forEach((name, i) => {
                if (this.includeInWeightDecay.includes(name)) {
                    const value = this.ENGINE.registeredVariables[name]
                    const newValue = tf.sub(
                        value,
                        tf.mul(this.learningRate, tf.mul(value, this.decayRate))
                    )
                    value.assign(newValue)
                }
            })

            super.applyGradients(variableGradients)
        })
    }
}

// class AdamW extends tf.AdamOptimizer {
//     constructor(
//         learningRate = 0.1,
//         beta1 = 0.9,
//         beta2 = 0.999,
//         epsilon = 1e-7,
//         decayRate = 1e-4,
//         includeInWeightDecay,
//         excludeFromWeightDecay
//     ) {
//         super(learningRate, beta1, beta2, epsilon)
//         this.ENGINE = tf.engine()
//         this.decayRate = decayRate
//         this.includeInWeightDecay = includeInWeightDecay
//         this.excludeFromWeightDecay = excludeFromWeightDecay
//         this.previousGrads = []
//         this.differences = []
//     }

//     applyGradients(variableGradients) {
//         tf.tidy(() => {
//             const variableNames = Array.isArray(variableGradients)
//                 ? variableGradients.map((v) => v.name)
//                 : Object.keys(variableGradients)

//             if (this.previousGrads.length > 0) {
//                 this.differences = variableNames
//                     .filter((x) => !this.previousGrads.includes(x))
//                     .concat(
//                         this.previousGrads.filter(
//                             (x) => !variableNames.includes(x)
//                         )
//                     )
//             }
//             this.previousGrads = variableNames

//             // Object.keys(this.accumulatedFirstMoment).forEach((i) => {
//             //     console.log(
//             //         'first shape: ',
//             //         i,
//             //         this.accumulatedFirstMoment[i].variable.shape
//             //     )
//             //     console.log(
//             //         'second shape: ',
//             //         i,
//             //         this.accumulatedSecondMoment[i].variable.shape
//             //     )
//             //     // tf.dispose([
//             //     //     this.accumulatedFirstMoment[i],
//             //     //     this.accumulatedSecondMoment[i]
//             //     // ])
//             //     // delete this.accumulatedFirstMoment[i]
//             //     // delete this.accumulatedSecondMoment[i]
//             // })

//             // const momentNames = this.accumulatedFirstMoment.map((i) => {
//             //     // tf.dispose([
//             //     //     this.accumulatedFirstMoment[i],
//             //     //     this.accumulatedSecondMoment[i]
//             //     // ])
//             //     // delete this.accumulatedFirstMoment[i]
//             //     // delete this.accumulatedSecondMoment[i]
//             //     return i.originalName.slice(0, -2)
//             // })

//             // console.log(variableNames)
//             // Update optimizer state only for layers present in current gradients
//             this.differences.map((name) => {
//                 console.log(name)
//                 console.log(
//                     Object.entries(this.accumulatedFirstMoment).originalName[
//                         name + '/m'
//                     ]
//                 )
//                 // if (!variableNames.includes(name)) {
//                 //     console.log(name)
//                 //     Object.keys(this.accumulatedFirstMoment).forEach((i) => {
//                 //         console.log(i.variable.originalName)
//                 //         if (i.originalName.includes(name + '/m')) {
//                 //             console.log(name)
//                 //             tf.dispose([
//                 //                 this.accumulatedFirstMoment[i],
//                 //                 this.accumulatedSecondMoment[i]
//                 //             ])
//                 //             delete this.accumulatedFirstMoment[i]
//                 //             delete this.accumulatedSecondMoment[i]
//                 //         }
//                 //     })
//                 // }
//             })

//             // console.log(this)

//             // momentNames2.map((moment) => {
//             //     if (!variableNames.includes(moment)) {
//             //         console.log(moment)
//             //     }
//             // })
//             // Object.keys(this.accumulatedFirstMoment).forEach((i) => {
//             //     for (const moment of this.accumulatedFirstMoment) {

//             //         if ()

//             //         console.log(moment.originalName.slice(0, -2))
//             //         if (
//             //             !variableNames.includes(
//             //                 moment.originalName.slice(0, -2)
//             //             )
//             //         ) {
//             //             console.log(moment.originalName.slice(0, -2))
//             //         }
//             //     }

//             // if (
//             //     !variableNames.includes(
//             //         this.accumulatedFirstMoment[i].originalName.replace(
//             //             /\/m$/,
//             //             ''
//             //         )
//             //     )
//             // ) {
//             //     console.log(this.accumulatedFirstMoment[i].originalName)
//             //     tf.dispose([
//             //         this.accumulatedFirstMoment[i],
//             //         this.accumulatedSecondMoment[i]
//             //     ])
//             //     delete this.accumulatedFirstMoment[i]
//             //     delete this.accumulatedSecondMoment[i]
//             // }
//             // })
//             // Object.keys(this.accumulatedFirstMoment).forEach((i) => {
//             //     if (
//             //         !variableNames.includes(
//             //             this.accumulatedFirstMoment[i].originalName
//             //         )
//             //     ) {
//             //         console.log(this.accumulatedFirstMoment[i].originalName)
//             //         tf.dispose([
//             //             this.accumulatedFirstMoment[i],
//             //             this.accumulatedSecondMoment[i]
//             //         ])
//             //         delete this.accumulatedFirstMoment[i]
//             //         delete this.accumulatedSecondMoment[i]
//             //     }
//             // })

//             // Apply weight decay
//             variableNames.forEach((name) => {
//                 if (this.includeInWeightDecay.includes(name)) {
//                     const value = this.ENGINE.registeredVariables[name]
//                     const newValue = tf.sub(
//                         value,
//                         tf.mul(this.learningRate, tf.mul(value, this.decayRate))
//                     )
//                     value.assign(newValue)
//                 }
//             })

//             super.applyGradients(variableGradients)
//         })
//     }
// }

function prepareAdamW(model, learningRate, beta1, beta2, epsilon, decayRate) {
    const includeInWeightDecay = []
    const excludeFromWeightDecay = []

    if (decayRate <= 0) {
        throw 'AdamW with a decayRate of 0 is just regular Adam. You should use `tf.train.adam` instead.'
    } else {
        model.getNamedWeights().forEach((v) => {
            const name = v.name.toLowerCase()
            if (
                name.includes('bias') ||
                name.includes('norm') ||
                name.includes('emb')
            ) {
                excludeFromWeightDecay.push(v.name)
            } else {
                includeInWeightDecay.push(v.name)
            }
        })
        return new AdamW(
            learningRate,
            beta1,
            beta2,
            epsilon,
            decayRate,
            includeInWeightDecay,
            excludeFromWeightDecay
        )
    }
}

// class Prodigy extends tf.train.Optimizer {
//     constructor(
//         learningRate = 1.0,
//         beta1 = 0.9,
//         beta2 = 0.999,
//         beta3 = null,
//         d0 = 1e-6,
//         dCoef = 1.0,
//         growthRate = Infinity,
//         weightDecay = 0.0,
//         weightDecouple = true,
//         fixedDecay = false,
//         biasCorrection = false,
//         safeguardWarmup = false,
//         epsilon = 1e-8
//     ) {
//         super()
//         this.learningRate = learningRate
//         this.beta1 = beta1
//         this.beta2 = beta2
//         this.beta3 = beta3 || Math.sqrt(beta2)
//         this.d0 = d0
//         this.dCoef = dCoef
//         this.growthRate = growthRate
//         this.weightDecay = weightDecay
//         this.weightDecouple = weightDecouple
//         this.fixedDecay = fixedDecay
//         this.biasCorrection = biasCorrection
//         this.safeguardWarmup = safeguardWarmup
//         this.epsilon = epsilon
//         this.d = d0
//         this.dMax = d0
//         this.step = 1
//     }

//     applyGradients(variableGradients) {
//         const beta1 = this.beta1
//         const beta2 = this.beta2
//         const beta3 = this.beta3
//         const biasCorrection1 = 1.0 - Math.pow(beta1, this.step)
//         const biasCorrection2Sq = Math.sqrt(1.0 - Math.pow(beta2, this.step))
//         const biasCorrection = this.biasCorrection
//             ? biasCorrection1 / biasCorrection2Sq
//             : 1.0
//         const dLr = (this.d * this.learningRate) / biasCorrection
//         let dNumerator = 0
//         let dDenom = 0

//         tf.tidy(() => {
//             variableGradients.forEach(({ name, tensor, gradients }) => {
//                 const variable = this.ENGINE.registeredVariables[name]
//                 const state = this.STATE[name] || {}
//                 const p0 = state.p0 || variable.clone()
//                 const expAvg = state.expAvg || tf.zerosLike(variable)
//                 const expAvgSq = state.expAvgSq || tf.zerosLike(variable)
//                 const s = state.s || tf.zerosLike(variable)

//                 dNumerator += tf
//                     .dot(gradients.flatten(), tf.sub(p0, variable).flatten())
//                     .mul((this.d / this.d0) * dLr)
//                     .dataSync()[0]

//                 expAvg.assign(
//                     expAvg.mul(beta1).add(gradients.mul(this.d * (1.0 - beta1)))
//                 )
//                 expAvgSq.assign(
//                     expAvgSq
//                         .mul(beta2)
//                         .addcmul(
//                             gradients,
//                             gradients,
//                             this.d * this.d * (1.0 - beta2)
//                         )
//                 )

//                 s.assign(
//                     s
//                         .mul(beta3)
//                         .add(
//                             gradients.mul(
//                                 (this.d / this.d0) *
//                                     (this.safeguardWarmup ? this.d : dLr)
//                             )
//                         )
//                 )

//                 dDenom += s.abs().sum().dataSync()[0]

//                 this.STATE[name] = { p0, expAvg, expAvgSq, s }
//             })
//         })

//         const dHat = (this.dCoef * dNumerator) / dDenom
//         if (this.d === this.d0) {
//             this.d = Math.max(this.d, dHat)
//         }
//         this.dMax = Math.max(this.dMax, dHat)
//         this.d = Math.min(this.dMax, this.d * this.growthRate)

//         tf.tidy(() => {
//             variableGradients.forEach(({ name, tensor, gradients }) => {
//                 const variable = this.ENGINE.registeredVariables[name]
//                 const { expAvg, expAvgSq } = this.STATE[name]

//                 const denom = expAvgSq.sqrt().add(this.d * this.epsilon)

//                 if (this.weightDecay !== 0) {
//                     if (this.weightDecouple) {
//                         variable.assign(
//                             variable.sub(variable.mul(this.weightDecay * dLr))
//                         )
//                     } else if (!this.fixedDecay) {
//                         gradients = gradients.add(
//                             variable.mul(this.weightDecay)
//                         )
//                     }
//                 }

//                 variable.assign(variable.sub(expAvg.div(denom).mul(dLr)))
//             })
//         })

//         this.step++
//     }

//     getConfig() {
//         return {
//             learningRate: this.learningRate,
//             beta1: this.beta1,
//             beta2: this.beta2,
//             beta3: this.beta3,
//             d0: this.d0,
//             dCoef: this.dCoef,
//             growthRate: this.growthRate,
//             weightDecay: this.weightDecay,
//             weightDecouple: this.weightDecouple,
//             fixedDecay: this.fixedDecay,
//             biasCorrection: this.biasCorrection,
//             safeguardWarmup: this.safeguardWarmup,
//             epsilon: this.epsilon
//         }
//     }
// }

const customOptimizers = {
    AdamW: (model, learningRate, beta1, beta2, epsilon, decayRate) =>
        prepareAdamW(model, learningRate, beta1, beta2, epsilon, decayRate)
}

export default customOptimizers
