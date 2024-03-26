function* constantScheduler(max) {
    while (true) {
        yield max
    }
}

function* cosineScheduler(max, min, totalIterations, modulation = 1) {
    let i = 0
    const range = max - min
    while (true) {
        let adjustedI = i / modulation

        let cosValue = Math.cos(
            (2 * Math.PI * adjustedI) / totalIterations + Math.PI
        )
        let currentValue = min + (range * (1 - cosValue)) / 2

        yield currentValue

        i = (i + 1) % (totalIterations * modulation)
    }
}

const schedulers = {
    constantScheduler: (max) => constantScheduler(max),
    cosineScheduler: (max, min, totalIterations, modulation) =>
        cosineScheduler(max, min, totalIterations, modulation)
}
export default schedulers
