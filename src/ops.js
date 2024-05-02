import * as tf from '@tensorflow/tfjs'

function customGather(inputs, indices) {
    const forward = (inputs, indices, save) => {
        indices = indices.cast('int32')
        const value = tf.gather(inputs, indices)
        save([value, indices])
        const gradFunc = (dy, saved) => {
            const [savedValue, savedIndices] = saved
            const inputsGrad = () => {
                const inputsShape = inputs.shape
                const flattenedIndices = savedIndices.reshape([-1])
                const updatesShape = savedIndices.shape.concat(
                    dy.shape.slice(1)
                )
                const updatedInputsGradValues = tf.scatterND(
                    flattenedIndices.expandDims(1),
                    dy.reshape(updatesShape),
                    inputsShape
                )
                return updatedInputsGradValues
            }
            const indicesGrad = () => {
                const result = tf.zerosLike(savedIndices, 'float32')
                return result
            }
            return [inputsGrad(), indicesGrad()]
        }
        return { value, gradFunc }
    }

    return tf.customGrad(forward)(inputs, indices.cast('float32'))
}

function subliminalSpace(start, end, dimensions) {
    const linspaceGrad = (dydy) => {
        const gradStart = tf.sum(dydy)
        const gradEnd = tf.sum(dydy)
        return [gradStart, gradEnd]
    }

    const customLinspace = tf.customGrad((start, end) => {
        const startScalar = start.dataSync()[0]
        const endScalar = end.dataSync()[0]
        const value = tf.linspace(startScalar, endScalar, dimensions)
        const gradFunc = (dy) => {
            const grads = linspaceGrad(dy)
            return grads
        }
        return { value, gradFunc }
    })

    return customLinspace(start, end)
}

function subliminalTopk(input, k) {
    const result = tf.topk(input, k)
    return { value: result.values, indices: result.indices }
}

// function subliminalTopk(input, k) {
//     const topkGrad = (dy, indices) => {
//         const inputShape = input.shape
//         const [batchSize, seqLen] = inputShape

//         const dyReshaped = tf.reshape(dy, [-1])

//         const indicesFlat = tf
//             .range(0, batchSize)
//             .flatten()
//             .mul(seqLen)
//             .add(indices.flatten())

//         const gradInput = tf.zeros(inputShape, input.dtype)
//         const updatedGradInput = gradInput.scatter(indicesFlat, dyReshaped, 1)

//         return updatedGradInput
//     }

//     const customTopk = tf.customGrad((input, save) => {
//         const { values, indices } = tf.topk(input, k)
//         save([indices])
//         const gradFunc = (dy, saved) => {
//             const [savedIndices] = saved
//             const gradInput = topkGrad(dy, savedIndices)
//             return gradInput
//         }
//         return { value: values, gradFunc }
//     })

//     const value = customTopk(input)
//     const indices = tf.topk(input, k).indices
//     return { value, indices }
// }

export default {
    subliminalSpace,
    subliminalTopk,
    customGather
}
