export default async function loadModelVersion(args) {
    const defaults = {
        version: 4,
        ...args
    }
    try {
        const module = await import(`./model.v${defaults.version}.js`)
        return new module.default(args)
    } catch (error) {
        console.error(
            `Failed to load model version ${defaults.version}:`,
            error
        )
        throw error
    }
}
