const focalLoss = (gamma = 2.0, alpha = 0.25) => {
    return (yTrue, yPred) => {
        // Clip predictions to prevent log(0) error
        const epsilon = 1e-7
        yPred = yPred.clipByValue(epsilon, 1 - epsilon)

        // Calculate focal loss components
        const crossEntropy = yTrue.mul(yPred.log()).mul(-1)
        const modulatingFactor = yTrue.mul(1 - yPred).pow(gamma)
        const alphaWeighted = yTrue.mul(alpha)

        // Combine components to get the final loss
        const focalLoss = alphaWeighted.mul(modulatingFactor).mul(crossEntropy)

        // Reduce the loss to a single scalar value
        return focalLoss.mean()
    }
}
