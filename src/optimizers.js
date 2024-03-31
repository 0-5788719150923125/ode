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

const customOptimizers = {
    AdamW: (model, learningRate, beta1, beta2, epsilon, decayRate) =>
        prepareAdamW(model, learningRate, beta1, beta2, epsilon, decayRate)
}

export default customOptimizers
