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

function initializePlot(width, height) {
    return new Array(height).fill('').map(() => new Array(width).fill(' '))
}

function updatePlot(plot, value, maxVal, minVal, width, height) {
    plot.forEach((line) => {
        for (let i = 0; i < width - 1; i++) {
            line[i] = line[i + 1]
        }
        line[width - 1] = ' '
    })

    const scaleFactor = height / (maxVal - minVal)
    let y = Math.floor((value - minVal) * scaleFactor)
    y = Math.max(0, Math.min(height - 1, y)) // Ensure y is within bounds

    plot[height - 1 - y].fill('*', width - 1, width) // Draw the new asterisk at the rightmost position
}

function printPlot(plot) {
    console.clear()
    console.log(plot.map((line) => line.join('')).join('\n'))
}

function plotCosineSchedulerRealTime(
    min,
    max,
    iterations,
    width = 75,
    height = 30,
    rate = 100
) {
    const generator = cosineScheduler(min, max, iterations)
    let plot = initializePlot(width, height)

    function plotNext() {
        const value = generator.next().value
        updatePlot(plot, value, max, min, width, height)
        printPlot(plot)

        setTimeout(plotNext, rate)
    }

    plotNext()
}

// Usage
plotCosineSchedulerRealTime(0, 1, 200, 75, 10, 100)
