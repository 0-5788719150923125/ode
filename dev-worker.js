import { trainModel } from './dev-train.js'

onmessage = async function (event) {
    await trainModel(event.data)
}
