class SchedulerBase {
    constructor(args) {
        this.args = args
        this.currentStep = 0
    }

    step() {
        throw new Error('step() method must be implemented in derived classes')
    }

    getConfig() {
        return {
            class: this.constructor.name
        }
    }
}

class ConstantScheduler extends SchedulerBase {
    constructor({ max, warmupSteps = 0 }) {
        super({ max, warmupSteps })
        this.max = max
        this.warmupSteps = warmupSteps
    }

    step() {
        if (this.currentStep < this.warmupSteps) {
            const t = this.currentStep / this.warmupSteps
            const lr = this.max * t
            this.currentStep++
            return lr
        }
        return this.max
    }

    getConfig() {
        return {
            ...super.getConfig(),
            max: this.max,
            warmupSteps: this.warmupSteps
        }
    }
}

class CosineScheduler extends SchedulerBase {
    constructor({ min, max, totalSteps, warmupSteps = 0, modulation = 1 }) {
        super({ min, max, totalSteps, warmupSteps, modulation })
        this.min = min
        this.max = max
        this.totalSteps = totalSteps + warmupSteps
        this.warmupSteps = warmupSteps
        this.modulation = modulation
        this.range = min - max
    }

    step() {
        if (this.currentStep < this.warmupSteps) {
            const t = this.currentStep / this.warmupSteps
            const lr = this.max * t
            this.currentStep++
            return lr
        }

        const adjustedI =
            (this.currentStep - this.warmupSteps) / this.modulation
        const cosValue = Math.cos(
            Math.PI *
                ((2 * (adjustedI % this.totalSteps)) / this.totalSteps - 1)
        )
        const currentValue = this.max + (this.range * (1 + cosValue)) / 2

        this.currentStep++
        return currentValue
    }

    getConfig() {
        return {
            ...super.getConfig(),
            min: this.min,
            max: this.max,
            totalSteps: this.totalSteps,
            warmupSteps: this.warmupSteps,
            modulation: this.modulation
        }
    }
}

class CosineWithRestartsScheduler extends SchedulerBase {
    constructor({ min, max, totalSteps, warmupSteps }) {
        super({ min, max, totalSteps, warmupSteps })
        this.min = min
        this.max = max
        this.totalSteps = totalSteps
        this.warmupSteps = warmupSteps
        this.cycleStep = 0
    }

    step() {
        if (this.currentStep < this.warmupSteps) {
            const t = this.currentStep / this.warmupSteps
            const lr = this.max * t
            this.currentStep++
            return lr
        }

        const t = this.cycleStep / this.totalSteps
        const cosineDecay = 0.5 * (1 + Math.cos(Math.PI * t))
        const lr = this.min + (this.max - this.min) * cosineDecay

        this.currentStep++
        this.cycleStep = (this.cycleStep + 1) % this.totalSteps
        return lr
    }

    getConfig() {
        return {
            ...super.getConfig(),
            min: this.min,
            max: this.max,
            totalSteps: this.totalSteps,
            warmupSteps: this.warmupSteps
        }
    }
}

const schedulers = {
    ConstantScheduler: (args) => new ConstantScheduler(args),
    CosineScheduler: (args) => new CosineScheduler(args),
    CosineWithRestartsScheduler: (args) => new CosineWithRestartsScheduler(args)
}

export default schedulers
