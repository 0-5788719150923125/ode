import * as tf from '@tensorflow/tfjs'

function customGather(inputs, indices) {
    const forward = (inputs, save) => {
        console.log('Forward inputs shape:', inputs.shape)
        console.log('Forward indices shape:', indices.shape)
        const value = tf.gather(inputs, indices.cast('int32'))
        save([value, indices])
        console.log('Forward value shape:', value.shape)
        const gradFunc = (dy, saved) => {
            const [savedValue, savedIndices] = saved
            console.log('Backward dy shape:', dy.shape)
            const inputsGrad = () => {
                const inputsShape = inputs.shape
                const indicesShape = savedIndices.shape
                const dyShape = dy.shape

                const inputsGradValues = tf.zerosLike(inputs)
                const flattenedIndices = savedIndices.reshape([-1])
                const updatesShape = savedIndices.shape.concat(
                    dy.shape.slice(1)
                )
                const updatedInputsGradValues = tf.scatterND(
                    flattenedIndices.expandDims(1),
                    dy.reshape(updatesShape),
                    inputsShape
                )

                console.log(
                    'Backward inputsGrad shape:',
                    updatedInputsGradValues.shape
                )
                return updatedInputsGradValues
            }
            const indicesGrad = () => {
                const result = tf.zerosLike(savedIndices, 'float32')
                console.log('Backward indicesGrad shape:', result.shape)
                return result
            }
            return [
                inputsGrad()
                // indicesGrad()
            ]
        }
        return { value, gradFunc }
    }

    return tf.customGrad(forward)(inputs)
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
    const topkGrad = (dy) => {
        const inputShape = input.shape
        const [batchSize, seqLen] = inputShape

        const dyReshaped = tf.reshape(dy, [-1])

        const indices = tf
            .range(0, batchSize)
            .flatten()
            .mul(seqLen)
            .add(
                tf
                    .topk(tf.reshape(input, [batchSize, seqLen]), k)
                    .indices.flatten()
            )

        const gradInput = tf.zeros(inputShape, input.dtype)
        const updatedGradInput = gradInput.scatter(indices, dyReshaped, 1)

        return updatedGradInput
    }

    let idx
    const customTopk = tf.customGrad((input) => {
        const { values, indices } = tf.topk(input, k)
        idx = tf.keep(indices)
        const gradFunc = (dy) => {
            const gradInput = topkGrad(dy)
            return [gradInput, null]
        }
        return { value: values, gradFunc }
    })

    const values = customTopk(input)
    return { values, indices: idx }
}

export default {
    subliminalSpace,
    subliminalTopk,
    customGather
}
