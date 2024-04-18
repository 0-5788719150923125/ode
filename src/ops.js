import * as tf from '@tensorflow/tfjs'

function subliminalSpace(start, end, dimensions) {
    const linspaceGrad = (dydy) => {
        const gradStart = tf.sum(dydy)
        const gradEnd = tf.sum(dydy)
        return [gradStart, gradEnd]
    }

    const customLinspace = tf.customGrad((start, end) => {
        const startScalar = start.dataSync()[0]
        const endScalar = end.dataSync()[0]
        const linspace = tf.linspace(startScalar, endScalar, dimensions)
        const value = linspace
        const gradFunc = (dy) => {
            const grads = linspaceGrad(dy)
            return grads
        }
        return { value, gradFunc }
    })

    return customLinspace(start, end)
}

const ops = {
    subliminalSpace
}
export default ops
