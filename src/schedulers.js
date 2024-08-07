function* constantScheduler(max) {
    while (true) {
        yield max
    }
}

function* cosineScheduler(start, end, totalIterations, modulation = 1) {
    let i = 0
    const range = end - start
    while (true) {
        // Adjust iteration for modulation
        let adjustedI = i / modulation

        // Calculate cosine value for cyclical annealing
        let cosValue = Math.cos(
            Math.PI *
                ((2 * (adjustedI % totalIterations)) / totalIterations - 1)
        )

        // Adjust current value based on cosine, equally applied at both ends
        let currentValue = start + (range * (1 + cosValue)) / 2

        yield currentValue

        // Increment iteration
        i++
    }
}

function* cosineWithRestartsScheduler(min, max, steps) {
    while (true) {
        for (let i = 0; i < steps; i++) {
            const t = i / steps
            const cosineDecay = 0.5 * (1 + Math.cos(Math.PI * t))
            const lr = min + (max - min) * cosineDecay
            yield lr
        }
    }
}

const schedulers = {
    constantScheduler: (max) => constantScheduler(max),
    cosineScheduler: (max, min, totalIterations, modulation) =>
        cosineScheduler(max, min, totalIterations, modulation),
    cosineWithRestartsScheduler: (min, max, totalIterations) =>
        cosineWithRestartsScheduler(min, max, totalIterations)
}
export default schedulers
