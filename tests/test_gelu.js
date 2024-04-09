import * as tf from '@tensorflow/tfjs'

// /**
//  * Gelu activation function
//  */
// export class Gelu extends Activation {
//     /** @nocollapse */
//     static readonly className = 'gelu';
//     /**
//      * Calculate the activation function.
//      *
//      * @param x Tensor.
//      * @returns a Tensor of the same shape as x
//      */
//     apply(x: Tensor): Tensor {
//       return tidy(() => {
//         return tfc.tidy(() => {
//           const sqrtTwo = Math.sqrt(2);
//           // Compute Φ(x) using the erf function
//           const cdf = tfc.mul(0.5, tfc.add(1, tfc.erf(tfc.div(x, sqrtTwo))));
//           // Compute GELU(x) = x * Φ(x)
//           return tfc.mul(x, cdf);
//         });
//       });
//     }
//   }
//   serialization.registerClass(Gelu);

//   /**
//    * GeluNew activation function
//    */
//   export class GeluNew extends Activation {
//     /** @nocollapse */
//     static readonly className = 'gelu_new';
//     /**
//      * Calculate the activation function.
//      *
//      * @param x Tensor.
//      * @returns a Tensor of the same shape as x
//      */
//     apply(x: Tensor): Tensor {
//       return tidy(() => {
//         return tfc.mul(
//           0.5,
//           tfc.add(
//               1,
//               tfc.tanh(
//                 tfc.mul(
//                   tfc.sqrt(tfc.div(2, Math.PI)),
//                   tfc.add(x, tfc.mul(0.044715, tfc.pow(x, 3)))
//                   )
//               )
//           )
//         );
//       });
//     }
//   }
//   serialization.registerClass(GeluNew);

// function compute(x) {
//     return tf.tidy(() => {
//         return tf.mul(
//             0.5,
//             tf.add(
//                 1,
//                 tf.tanh(
//                     tf.mul(
//                         tf.sqrt(tf.div(2, Math.PI)),
//                         tf.add(x, tf.mul(0.044715, tf.pow(x, 3)))
//                     )
//                 )
//             )
//         )
//     })
// }

// const tensor = tf.tensor3d([
//     [
//         [0, 1, 3, 9],
//         [0, 1, 3, 9]
//     ]
// ])

// console.log(compute(tensor).arraySync())

console.log(tf.elu(0.3).arraySync())
