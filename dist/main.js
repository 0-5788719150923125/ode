async function $c4cf1330f3877361$export$2e2bcd8739ae039(args) {
    const defaults = {
        version: 4,
        ...args
    }
    const maxVersion = 4
    const models = []
    try {
        const module1 = await import(`./model.v$1.js`)
        for (let i = 0; i <= maxVersion; i++) {
            const module = await import(`./model.v${i}.js`)
            models.push(module)
        }
        return new models[Number(defaults.version)].default(args)
    } catch (error) {
        console.error(
            `Failed to load model version ${defaults.version}:`,
            error
        )
        throw error
    }
}

var $cf838c15c8b009ba$export$2e2bcd8739ae039 =
    (0, $c4cf1330f3877361$export$2e2bcd8739ae039)

export { $cf838c15c8b009ba$export$2e2bcd8739ae039 as default }
//# sourceMappingURL=main.js.map
