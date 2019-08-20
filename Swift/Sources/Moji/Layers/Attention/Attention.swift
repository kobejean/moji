import TensorFlow


// MARK: Attention

public struct Attention: ParameterlessLayer {
    
    // MARK: Properties
    
    @noDerivative public let dropout: Dropout<Float>
    @noDerivative public let scale: Tensor<Float>
    
    
    // MARK: Initializer
    
    public init(size: Int, dropProbability: Double = 0.2) {
        scale = Tensor(sqrt(Float(size)))
        dropout = Dropout<Float>(probability: dropProbability)
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return callAsFunction(input, mask: nil)
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>, mask: Tensor<Float>? = nil) -> Tensor<Float> {
        let qkv = AttentionQueryKeyValue.make(input)
        return attend(query: qkv.query, key: qkv.key, value: qkv.value, mask: mask)
    }
    
    @differentiable(wrt: (self, input))
    public func callAsFunction(_ input: Tensor<Float>, state: inout AttentionContext, mask: Tensor<Float>? = nil) -> Tensor<Float> {
        let qkv = AttentionQueryKeyValue.make(input)
        state = AttentionContext.make(
            key: state.key.concatenated(with: qkv.key, alongAxis: 1),
            value: state.value.concatenated(with: qkv.value, alongAxis: 1)
        )
        return attend(query: qkv.query, key: state.key, value: state.value, mask: mask)
    }
    
    @differentiable(wrt: (query, key, value))
    public func attend(query: Tensor<Float>, key: Tensor<Float>, value: Tensor<Float>, mask: Tensor<Float>? = nil) -> Tensor<Float> {
        let logits = batchedMatmul(query, key, adjointRight: true) / scale
        let score = maskedScore(logits, mask: mask)
        return batchedMatmul(dropout(score), value)
    }
    
    
    // MARK: Internal Methods
    
    @inlinable
    @differentiable(wrt: logits)
    internal func maskedScore(_ logits: Tensor<Float>, mask: Tensor<Float>? = nil) -> Tensor<Float> {
        guard let mask = mask else { return softmax(logits) }
        return maskedSoftmax(logits, mask: mask)
    }
}


// MARK: AttentionQueryKeyValue

public struct AttentionQueryKeyValue: Differentiable {
    public var query: Tensor<Float>
    public var key: Tensor<Float>
    public var value: Tensor<Float>
    
    @differentiable(wrt: input, vjp: _vjpMake)
    public static func make(_ input: Tensor<Float>) -> Self {
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
        return Self(query: query, key: key, value: value)
    }
    
    @inlinable
    internal static func _vjpMake(_ input: Tensor<Float>) -> (Self, (Self.TangentVector) -> Tensor<Float>) {
        let value = Self.make(input)
        return (value, { seed in
            return Raw.concatV2([seed.query, seed.key, seed.value], axis: Tensor<Int32>(2))
        })
    }
}


// MARK: AttentionContext

public struct AttentionContext: Differentiable {
    public var key: Tensor<Float>
    public var value: Tensor<Float>
    
    @differentiable(wrt: (key, value), vjp: _vjpMake)
    public static func make(key: Tensor<Float>, value: Tensor<Float>) -> Self {
        return Self(key: key, value: value)
    }

    @usableFromInline
    internal static func _vjpMake(key: Tensor<Float>, value: Tensor<Float>)
        -> (Self, (Self.TangentVector) -> (Tensor<Float>, Tensor<Float>)) {
        let result = Self(key: key, value: value)
        return (result, { seed in (seed.key, seed.value) })
    }
}
