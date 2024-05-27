import schedulers from '../src/schedulers.js'

function initializePlot(width, height) {
    return new Array(height).fill('').map(() => new Array(width).fill(' '))
}

function updatePlot(plot, value, maxVal, minVal, width, height) {
    // Shift everything to the left
    plot.forEach((line) => {
        for (let i = 0; i < width - 1; i++) {
            line[i] = line[i + 1]
        }
        line[width - 1] = ' '
    })

    // Scale factor to map value to plot coordinates
    const scaleFactor = height / (maxVal - minVal)
    let y = Math.floor((value - minVal) * scaleFactor)

    // Ensure y is within bounds
    y = height - 1 - Math.max(0, Math.min(height - 1, y))

    // Update the plot with the new value at the correct position
    plot[y].fill('*', width - 1, width)
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
    rate = 100,
    modulation = 1
) {
    const generator = schedulers.cosineWithRestartsScheduler(
        min,
        max,
        iterations,
        modulation
    )
    let plot = initializePlot(width, height)

    function plotNext() {
        const value = generator.next().value
        updatePlot(plot, value, max, min, width, height)
        printPlot(plot)
        // console.log(value)

        setTimeout(plotNext, rate)
    }

    plotNext()
}

// Usage
const modulation = 0.666
plotCosineSchedulerRealTime(0, 1, 333, 75, 20, 100, modulation)
// plotCosineSchedulerRealTime(1, 0, 1000, 75, 20, 100)
