export function applyWeightDecay(
    variable,
    gradient,
    name,
    learningRate,
    weightDecay = 0.0,
    weightDecouple = true,
    fixedDecay = false
) {
    if (weightDecay === 0 && shouldExcludeFromWeightDecay(name)) {
        return gradient
    }

    if (weightDecouple) {
        variable.assign(variable.sub(variable.mul(weightDecay * learningRate)))
    } else if (fixedDecay) {
        gradient = gradient.add(variable.mul(weightDecay))
    }

    return gradient
}

function shouldExcludeFromWeightDecay(name) {
    const lowerCaseName = name.toLowerCase()
    return (
        lowerCaseName.includes('norm') ||
        lowerCaseName.includes('emb') ||
        lowerCaseName.includes('bias')
    )
}
