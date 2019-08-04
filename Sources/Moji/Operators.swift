//
//  Operators.swift
//  
//
//  Created by Jean Flaherty on 8/3/19.
//

import TensorFlow


/// Performs batched matrix multiplication of two tensors. The last two axes of each tensor
/// are treated as the matrix axes; all others are treated as batch axes.
@differentiable(
    wrt: (left, right),
    vjp: _vjpBatchedMatmul
    where Scalar : Differentiable & TensorFlowFloatingPoint
)
public func batchedMatmul<Scalar : Numeric>(
    _ left: Tensor<Scalar>,
    _ right: Tensor<Scalar>,
    adjointLeft: Bool = false,
    adjointRight: Bool = false
) -> Tensor<Scalar> {
    return Raw.batchMatMul(left, right, adjX: adjointLeft, adjY: adjointRight)
}

@inlinable
internal func _vjpBatchedMatmul<Scalar : Differentiable & TensorFlowFloatingPoint>(
    _ left: Tensor<Scalar>,
    _ right: Tensor<Scalar>,
    adjointLeft: Bool,
    adjointRight: Bool
) -> (Tensor<Scalar>, (Tensor<Scalar>) -> (Tensor<Scalar>, Tensor<Scalar>)) {
    let value = batchedMatmul(left, right, adjointLeft: adjointLeft, adjointRight: adjointRight)
    return (value, { v in
        if !adjointLeft {
            if !adjointRight {
                return (
                    batchedMatmul(v, right, adjointLeft: false, adjointRight: true),
                    batchedMatmul(left, v, adjointLeft: true, adjointRight: false))
            } else {
                return (
                    batchedMatmul(v, right, adjointLeft: false, adjointRight: false),
                    batchedMatmul(v, left, adjointLeft: true, adjointRight: false))
            }
        } else {
            if !adjointRight {
                return (
                    batchedMatmul(right, v, adjointLeft: false, adjointRight: true),
                    batchedMatmul(left, v, adjointLeft: false, adjointRight: false))
            } else {
                return (
                    batchedMatmul(right, v, adjointLeft: true, adjointRight: true),
                    batchedMatmul(v, left, adjointLeft: true, adjointRight: true))
            }
        }
    })
}

@differentiable(wrt: logits, vjp: _vjpMaskedSoftmax)
public func maskedSoftmax<T: TensorFlowFloatingPoint>(_ logits: Tensor<T>, mask: Tensor<T>) -> Tensor<T> {
    let stable = logits - logits.max(alongAxes: -1)
    let exponentiated = exp(stable)
    let masked = exponentiated * mask
    let summed = masked.sum(alongAxes: -1)
    return masked / summed
}

public func _vjpMaskedSoftmax<T: TensorFlowFloatingPoint>(logits: Tensor<T>, mask: Tensor<T>)
-> (Tensor<T>, (Tensor<T>.TangentVector) -> Tensor<T>) {
    let y = maskedSoftmax(logits, mask: mask)
    return (y, { seed in (seed - (seed * y).sum(alongAxes: -1)) * y})
}
