import APTx from './activations/APTx.js'
import Laplace from './activations/Laplace.js'
import SERF from './activations/SERF.js'
import Snake from './activations/Snake.js'

export default {
    APTx: new APTx(),
    Laplace: new Laplace(),
    SERF: new SERF(),
    Snake: new Snake()
}
