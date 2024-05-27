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

function* cosineWithRestartsScheduler(start, end, totalIterations) {
    let i = 0
    const range = end - start
    while (true) {
        // Calculate cosine value for cyclical annealing
        let cosValue = Math.cos(
            Math.PI * ((2 * (i % totalIterations)) / totalIterations - 1)
        )

        // Adjust current value based on cosine, equally applied at both ends
        let currentValue = start + (range * (1 + cosValue)) / 2

        yield currentValue

        // Increment iteration
        i++

        // Hard restart when the total iterations are reached
        if (i % totalIterations === 0) {
            i = 0
        }
    }
}

const schedulers = {
    constantScheduler: (max) => constantScheduler(max),
    cosineScheduler: (max, min, totalIterations, modulation) =>
        cosineScheduler(max, min, totalIterations, modulation),
    cosineWithRestartsScheduler: (max, min, totalIterations) =>
        cosineWithRestartsScheduler(max, min, totalIterations)
}
export default schedulers
