import AdamG from './optimizers/AdamG.js'
import AdamW from './optimizers/AdamW.js'
import GrokFast from './optimizers/GrokFast.js'
import Lamb from './optimizers/Lamb.js'
import Lion from './optimizers/Lion.js'
import Prodigy from './optimizers/Prodigy.js'
import Signum from './optimizers/Signum.js'

export default {
    AdamW: (config) => new AdamW(config),
    AdamG: (config) => new AdamG(config),
    GrokFast: (config) => new GrokFast(config),
    Lamb: (config) => new Lamb(config),
    Lion: (config) => new Lion(config),
    Prodigy: (config) => new Prodigy(config),
    Signum: (config) => new Signum(config)
}
