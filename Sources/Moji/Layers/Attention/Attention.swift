import TensorFlow

// MARK: Attention

public struct Attention: Layer {
    // MARK: Properties
    var wqkv: TimeDistributed
    @noDerivative public let dropout: Dropout<Float>
    @noDerivative public let scale: Tensor<Float>
    @noDerivative public let causal: Bool
    
    // MARK: Initializer
    public init(size: Int, causal: Bool = false, dropProbability: Double = 0.2) {
        scale = Tensor(sqrt(Float(size)))
        dropout = Dropout<Float>(probability: dropProbability)
        wqkv = TimeDistributed(Dense<Float>(inputSize: size, outputSize: size * 3, activation: identity))
        self.causal = causal
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let qkvProjected = wqkv(input)
        let qkv = AttentionQueryKeyValue.make(qkvProjected)
        return attend(query: qkv.query, key: qkv.key, value: qkv.value)
    }
    
    @differentiable(wrt: (self, input))
    public func callAsFunction(_ input: Tensor<Float>, state: inout AttentionContext) -> Tensor<Float> {
        let qkvProjected = wqkv(input)
        let qkv = AttentionQueryKeyValue.make(qkvProjected)
        state = AttentionContext.make(
            key: state.key.concatenated(with: qkv.key, alongAxis: 1),
            value: state.value.concatenated(with: qkv.value, alongAxis: 1)
        )
        return attend(query: qkv.query, key: state.key, value: state.value)
    }
    
    @differentiable
    public func attend(query: Tensor<Float>, key: Tensor<Float>, value: Tensor<Float>) -> Tensor<Float>{
        let logits = batchedMatmul(query, key, adjointRight: true) / scale
        let score = maskedScore(logits)
        print("score", score)
        return batchedMatmul(dropout(score), value)
    }
    
    
    // MARK: Internal Methods
    
    @inlinable
    @differentiable(wrt: logits)
    internal func maskedScore(_ logits: Tensor<Float>) -> Tensor<Float> {
        guard causal else { return softmax(logits) }
        let (queryTimeSteps, keyTimeSteps) = (Tensor(Int32(logits.shape[1])), Tensor(Int32(logits.shape[2])))
        let mask = attentionMask(queryTimeSteps, keyTimeSteps)
        return maskedSoftmax(logits, mask: mask)
    }
    
    @inlinable
    @_semantics("autodiff.nonvarying")
    internal func attentionMask(_ queryTimeSteps: Tensor<Int32>, _ keyTimeSteps: Tensor<Int32>) -> Tensor<Float> {
        let currentTimeStep = Tensor<Int32>(0)
        let delta = Tensor<Int32>(1)
        let queryTimeStep = Raw.range(start: -queryTimeSteps, limit: currentTimeStep, delta: delta).expandingShape(at: -1)
        let keyTimeStep = Raw.range(start: -keyTimeSteps, limit: currentTimeStep, delta: delta)
        let mask = queryTimeStep .>= keyTimeStep
        return Raw.cast(mask)
    }
}


// MARK: AttentionQueryKeyValue

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
    
    @differentiable(wrt: input, vjp: _vjpMakeSplit)
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
    internal static func _vjpMakeSplit(_ input: Tensor<Float>)
        -> (Self, (Self.TangentVector) -> Tensor<Float>) {
        let value = Self.make(input)
        return (value, { seed in
            return Raw.concatV2([seed.query, seed.key, seed.value], axis: Tensor<Int32>(2))
        })
    }
}

// MARK: AttentionContext

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
