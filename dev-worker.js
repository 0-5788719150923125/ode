import { trainModel } from './dev-engine.js'

onmessage = async function (event) {
    if (event.command === 'train') await trainModel()
}
