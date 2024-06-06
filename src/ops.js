import * as tf from '@tensorflow/tfjs'

// Helper function to generate Gumbel noise
function sampleGumbel(shape, epsilon = 1e-8) {
    const uniform = tf.randomUniform(shape, 0, 1)
    return tf.neg(tf.log(tf.neg(tf.log(uniform.add(epsilon)))))
}

// Helper function to apply Gumbel-Softmax trick
function gumbelSoftmax(logits, temperature = 1.0) {
    const gumbelNoise = sampleGumbel(logits.shape)
    return tf.softmax(logits.add(gumbelNoise).div(temperature))
}

function reduceTimeStepsWithFFT(tensor, reducedTimeSteps) {
    const [batchSize, timeSteps, features] = tensor.shape

    // Custom gradient definition
    return tf.customGrad((tensor) => {
        // Apply FFT to the time dimension
        const fftTensor = tf.spectral.fft(
            tf.complex(tensor, tf.zerosLike(tensor))
        )

        // Truncate the high-frequency components
        const truncatedFFT = tf.slice(
            fftTensor,
            [0, 0, 0],
            [batchSize, reducedTimeSteps, features]
        )

        // Apply inverse FFT to get back to the time domain
        const ifftTensor = tf.spectral.ifft(truncatedFFT)

        // Return the real part of the result
        const realOutput = tf.real(ifftTensor)

        // Define the gradient function
        const gradFunc = (dy) => {
            // Calculate the gradient with respect to the input tensor
            const gradReal = tf.complex(dy, tf.zerosLike(dy))
            const gradIFFT = tf.spectral.fft(gradReal)

            // Pad the real and imaginary parts separately
            const realPart = tf.real(gradIFFT)
            const imagPart = tf.imag(gradIFFT)

            const padding = [
                [0, 0],
                [0, timeSteps - reducedTimeSteps],
                [0, 0]
            ]
            const paddedReal = tf.pad(realPart, padding)
            const paddedImag = tf.pad(imagPart, padding)

            // Combine the padded real and imaginary parts into a complex tensor
            const paddedGradIFFT = tf.complex(paddedReal, paddedImag)

            // Apply inverse FFT to the padded gradient tensor
            const gradFFT = tf.spectral.ifft(paddedGradIFFT)

            // Return the real part of the gradient
            return tf.real(gradFFT)
        }

        return { value: realOutput, gradFunc }
    })(tensor)
}

function reduceTimeStepsWithActivation(tensor, activationFunction, threshold) {
    return tf.tidy(() => {
        // Apply the activation function to the input tensor
        const activatedTensor = activationFunction(tensor)

        // Average the values along the feature dimension
        const averagedTensor = tf.mean(activatedTensor, -1)

        // Create a mask based on the threshold
        const mask = averagedTensor.greater(threshold)

        // Expand the mask dimensions to match the input tensor shape
        const expandedMask = mask.expandDims(-1)

        // Select the timesteps based on the mask
        const selectedTensor = tf.where(
            expandedMask,
            tensor,
            tf.zerosLike(tensor)
        )

        // Compress the selected tensor by removing the zero timesteps
        const compressedTensor = tf.compress(selectedTensor, 1, 2)

        return compressedTensor
    })
}

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

function sparseMixtureOfExpertsGrad(inputs, gatingScores, experts, topK) {
    const backward = (dy, saved) => {
        const [inputs, gatingScores, selectedExpertsTensor, outputs] = saved

        let gatingGrads = tf.zerosLike(gatingScores)
        let lastStepGatingGrads = tf.zeros([
            gatingScores.shape[0],
            experts.length
        ])

        for (let i = 0; i < inputs.shape[0]; i++) {
            const selectedExperts = selectedExpertsTensor.arraySync()[i]
            selectedExperts.map((index) => {
                const expertGrad = tf.ones([1, 1])
                const indices = tf.tensor1d([index], 'int32')
                lastStepGatingGrads = lastStepGatingGrads.add(
                    tf.oneHot(indices, experts.length).mul(expertGrad)
                )
            })
        }

        const reshapedLastStepGatingGrads = lastStepGatingGrads.reshape([
            gatingScores.shape[0],
            1,
            experts.length
        ])
        const lastStepSlice = gatingGrads.slice(
            [0, gatingScores.shape[1] - 1, 0],
            [gatingScores.shape[0], 1, experts.length]
        )
        gatingGrads = gatingGrads.add(
            lastStepSlice.sub(lastStepSlice).add(reshapedLastStepGatingGrads)
        )

        return [dy, gatingGrads]
    }

    const forward = tf.customGrad((inputs, gatingScores, save) => {
        const batchOutputs = []
        const selectedExpertsArray = []

        for (let i = 0; i < inputs.shape[0]; i++) {
            const batchInputs = inputs.slice([i, 0, 0], [1, -1, -1])
            const batchGatingScores = gatingScores.slice([i, 0, 0], [1, -1, -1])

            const { indices: topKIndices } = tf.topk(
                batchGatingScores.reshape([-1, experts.length]),
                topK
            )
            const selectedExperts = topKIndices.arraySync()[inputs.shape[1] - 1]
            selectedExpertsArray.push(selectedExperts)

            const expertOutputs = selectedExperts.map((index) => {
                return experts[index].apply(batchInputs)
            })

            const batchOutput = expertOutputs.reduce((acc, curr) =>
                acc.add(curr)
            )
            batchOutputs.push(batchOutput)
        }

        const selectedExpertsTensor = tf.tensor2d(
            selectedExpertsArray,
            [inputs.shape[0], topK],
            'int32'
        )

        const outputs = tf.concat(batchOutputs, 0)

        save([inputs, gatingScores, selectedExpertsTensor, outputs])

        return {
            value: outputs,
            gradFunc: (dy, saved) => {
                return backward(dy, saved)
            }
        }
    })

    return forward(inputs, gatingScores)
}

export default {
    subliminalSpace,
    subliminalTopk,
    customGather,
    sparseMixtureOfExpertsGrad,
    gumbelSoftmax,
    reduceTimeStepsWithFFT,
    reduceTimeStepsWithActivation
}
