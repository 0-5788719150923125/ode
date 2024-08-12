export function shouldExcludeFromWeightDecay(name) {
    const lowerCaseName = name.toLowerCase()
    return (
        lowerCaseName.includes('norm') ||
        lowerCaseName.includes('emb') ||
        lowerCaseName.includes('bias')
    )
}
