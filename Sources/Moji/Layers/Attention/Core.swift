import TensorFlow

// MARK: AttentionQueryKeyValue Definition

public struct AttentionQueryKeyValue: Differentiable {
    public var query: Tensor<Float>
    public var key: Tensor<Float>
    public var value: Tensor<Float>
    
    @differentiable(vjp: _vjpMake)
    public static func make(query: Tensor<Float>, key: Tensor<Float>, value: Tensor<Float>) -> Self {
        return Self(query: query, key: key, value: value)
    }

    public static func _vjpMake(query: Tensor<Float>, key: Tensor<Float>, value: Tensor<Float>) -> (Self, (Self.TangentVector) -> (Tensor<Float>, Tensor<Float>, Tensor<Float>)) {
        let result = Self(query: query, key: key, value: value)
        return (result, { seed in (seed.query, seed.key, seed.value) })
    }
    
    @differentiable(wrt: input, vjp: _vjpSplit)
    public static func split(_ input: Tensor<Float>) -> Self {
        let (batches, timeSteps, features) = (
            input.shape[0], input.shape[1], input.shape[2] / 3)
        let query = input.slice(
            lowerBounds: [0, 0, 0],
            upperBounds: [batches, timeSteps, features])
        let key = input.slice(
            lowerBounds: [0, 0, features],
            upperBounds: [batches, timeSteps, 2 * features])
        let value = input.slice(
            lowerBounds: [0, 0, 2 * features],
            upperBounds: [batches, timeSteps, 3 * features])
        return Self.make(query: query, key: key, value: value)
    }
    
    @inlinable
    internal static func _vjpSplit(_ input: Tensor<Float>)
        -> (Self, (Self.TangentVector) -> Tensor<Float>) {
        let value = Self.split(input)
        return (value, { seed in
            return Raw.concatV2([seed.query, seed.key, seed.value], axis: Tensor<Int32>(2))
        })
    }
}

// MARK: AttentionContext Definition

public struct AttentionContext: Differentiable {
    var key: Tensor<Float>
    var value: Tensor<Float>
    
    @differentiable(wrt: (key, value), vjp: _vjpMakeContext)
    static func make(key: Tensor<Float>, value: Tensor<Float>)
        -> Self {
            return Self(key: key, value: value)
    }

    @usableFromInline
    internal static func _vjpMakeContext(key: Tensor<Float>, value: Tensor<Float>)
        -> (Self, (Self.TangentVector) -> (Tensor<Float>, Tensor<Float>)) {
        let result = Self(key: key, value: value)
        return (result, { seed in (seed.key, seed.value) })
    }
}
