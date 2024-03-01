import { trainModel } from './dev-engine.js'

onmessage = async function (event) {
    await trainModel('webgl')
}
